from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import utils
import time
from catSNN import spikeLayer, load_model, max_weight, normalize_weight, SpikeDataset ,fuse_bn_recursively, fuse_module
from models.vgg_onlyt_vr_cifar100 import VGG_o_,CatVGG_o
import catSNN
import catCuda
T_reduce = 16
timestep = 20
timestep_f = 20.0
#f_name = 'neuron_100_trysoa.npz'
f_name = 'neuron_50.npz'
min_1 = 0
max_1 = T_reduce/timestep

def transfer_model(src, dst, quantize_bit=32):
    src_dict = src.state_dict()
    dst_dict = dst.state_dict()
    reshape_dict = {}
    for (k, v) in src_dict.items():
        if k in dst_dict.keys():
            #print(k)
            if 'weight' in k:
                #print("True")
                reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape)*max_1)
            else:
                reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))
    reshape_dict['features.6.weight'] = dst_dict['features.6.weight']*max_1
    reshape_dict['features.14.weight'] = dst_dict['features.14.weight']*max_1
    reshape_dict['features.25.weight'] = dst_dict['features.25.weight']*max_1
    reshape_dict['features.36.weight'] = dst_dict['features.36.weight']*max_1
    reshape_dict['features.47.weight'] = dst_dict['features.47.weight']*max_1
    #reshape_dict['classifier1.weight'] = dst_dict['classifier1.weight']*max_1
    #reshape_dict['classifier2.weight'] = dst_dict['classifier2.weight']*max_1
    dst.load_state_dict(reshape_dict, strict=False)


class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)
class AddQuantization_new(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        x_origin = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)
        #[0, 1/5, 2/5, 3/5]
        
        x_origin_plus_1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep+1)), timestep+1),min=min_1, max=T_reduce/(timestep+1))
        my_ones = torch.ones(x_origin_plus_1.shape[0],x_origin_plus_1.shape[1],x_origin_plus_1.shape[2])
        
        for i in range(1,T_reduce+1):
            x_origin_plus_1 = torch.where(x_origin_plus_1 == i*T_reduce / (T_reduce * (timestep + 1)), i*my_ones * max_1 / T_reduce, x_origin_plus_1)
        #[0, 1/6, 2/6, 3/6]
        
        x_origin_minus_1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep-1)), timestep-1),min=min_1, max=T_reduce/(timestep-1))
        my_ones = torch.ones(x_origin_minus_1.shape[0],x_origin_minus_1.shape[1],x_origin_minus_1.shape[2])
        for i in range(1,T_reduce+1):
            x_origin_minus_1 = torch.where(x_origin_minus_1 == i*T_reduce / (T_reduce * (timestep - 1)), i*my_ones * max_1 / T_reduce, x_origin_minus_1)
        #[0, 1/4, 2/4, 3/4]
    
        x = torch.cat((x_origin, x_origin_plus_1,x_origin_minus_1,x_origin,x_origin_plus_1,x_origin_minus_1), 0)
        return x

def change_shape(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def Initialize_trainable_pooling(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def test_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=T_reduce, metavar='N',
                        help='SNN time window')
    parser.add_argument('--k', type=int, default=50, metavar='N',
                        help='Data augmentation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
#9221
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])

    trainset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
       
    for i in range(args.k):

        im_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR100(root='./data', train=True, download=True, transform=im_aug)

    for i in range(args.k):
        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR100(root='./data', train=True, download=True, transform=im_aug)
    
    train_loader = torch.utils.data.DataLoader(
       trainset, batch_size=256+512, shuffle=True)

    testset = datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=15*8, shuffle=False)


    snn_dataset = SpikeDataset(testset, T = args.T,theta = max_1-0.001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=100, shuffle=False)

    
    model = VGG_o_('VGG16', clamp_max=1,bias =True).to(device)
    #vgg16_largeinput_NIPS_16_20_
    #cifar10_0_5_ASG_100_08_0316_100_1_full_input
    model.load_state_dict(torch.load("../../pretrain_weight/cifar100/vgg16_largeinput_NIPS_8_10_cifar100.pt"), strict=False)
    snn_model = CatVGG_o('VGG16', args.T,bias =True).to(device)

    

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    test(model, device, test_loader)

    correct_ = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader_)
        correct = test(model, device, test_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model.state_dict(), "vgg16_largeinput_NIPS_8_10_cifar100.pt")

        scheduler.step()
    
    model = fuse_bn_recursively(model)
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #for param_tensor in snn_model.state_dict():
    #    print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())

    transfer_model(model, snn_model)
    #with torch.no_grad():
    #    normalize_weight(snn_model.features, quantize_bit=32)
    test_(snn_model, device, snn_loader)
    #with torch.no_grad():
    #    normalize_weight(snn_model.features, quantize_bit=8)

    """
    model_dict = snn_model_training.state_dict()
    pre_dict = snn_model.state_dict()

    reshape_dict = {}
    reshape_dict['conv1.weight'] = nn.Parameter(pre_dict['features.0.weight'],requires_grad=False)
    reshape_dict['conv1.bias'] = nn.Parameter(pre_dict['features.0.bias'],requires_grad=False)

    reshape_dict['conv2.weight'] = nn.Parameter(pre_dict['features.3.weight'],requires_grad=False)
    reshape_dict['conv2.bias'] = nn.Parameter(pre_dict['features.3.bias'],requires_grad=False)

    reshape_dict['conv3.weight'] = nn.Parameter(pre_dict['features.8.weight'],requires_grad=False)
    reshape_dict['conv3.bias'] = nn.Parameter(pre_dict['features.8.bias'],requires_grad=False)

    reshape_dict['conv4.weight'] = nn.Parameter(pre_dict['features.11.weight'],requires_grad=False)
    reshape_dict['conv4.bias'] = nn.Parameter(pre_dict['features.11.bias'],requires_grad=False)

    reshape_dict['conv5.weight'] = nn.Parameter(pre_dict['features.16.weight'],requires_grad=False)
    reshape_dict['conv5.bias'] = nn.Parameter(pre_dict['features.16.bias'],requires_grad=False)

    reshape_dict['conv6.weight'] = nn.Parameter(pre_dict['features.19.weight'],requires_grad=False)
    reshape_dict['conv6.bias'] = nn.Parameter(pre_dict['features.19.bias'],requires_grad=False)

    reshape_dict['conv7.weight'] = nn.Parameter(pre_dict['features.24.weight'],requires_grad=False)
    reshape_dict['conv7.bias'] = nn.Parameter(pre_dict['features.24.bias'],requires_grad=False)

    reshape_dict['classifier1.weight']=nn.Parameter(pre_dict['classifier1.weight'],requires_grad=False)
    reshape_dict['classifier1.bias']=nn.Parameter(pre_dict['classifier1.bias'],requires_grad=False)

    model_dict.update(reshape_dict)
    snn_model_training.load_state_dict(model_dict)
    test_snn(snn_model_training, device, snn_loader)
    """

if __name__ == '__main__':
    main()
