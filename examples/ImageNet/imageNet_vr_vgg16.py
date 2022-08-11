from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
from torch.utils.data import Subset
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, load_model, max_weight, normalize_weight, SpikeDataset , fuse_bn_recursively
#from utils import to_tensor

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.models as models

T_reduce = 32
timestep = 40
timestep_f = 40
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
    reshape_dict['features.4.weight'] = dst_dict['features.4.weight']*max_1
    reshape_dict['features.10.weight'] = dst_dict['features.10.weight']*max_1
    reshape_dict['features.18.weight'] = dst_dict['features.18.weight']*max_1
    reshape_dict['features.26.weight'] = dst_dict['features.26.weight']*max_1
    reshape_dict['features.34.weight'] = dst_dict['features.34.weight']*max_1

    dst.load_state_dict(reshape_dict, strict=False)

class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def quantize_to_bit(x, nbit):
    if nbit == 32:
        return x
    return torch.mul(torch.round(torch.div(x, 2.0**(1-nbit))), 2.0**(1-nbit))

def data_loader(batch_size=128, workers=1, pin_memory=True):
    traindir = os.path.join('../../../../ImageNet/imagenet_raw/train')
    valdir = os.path.join('../../../../ImageNet/imagenet_raw/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            AddQuantization()
            #normalize
        ])
    )
    train_dataset1 = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            AddQuantization()
            #normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            #transforms.Resize(480),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AddQuantization()
            #normalize
        ])
    )
    #val_dataset_100 = Subset(val_dataset, range(0,15000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=60,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    num_training_samples = 10 
    train_sampler = SubsetRandomSampler(torch.arange(1000, 1100))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        #sampler=train_sampler
    )
    
    return train_loader, val_loader, val_dataset

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #onehot = torch.nn.functional.one_hot(target, 10)
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


def change_shape(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)
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
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    #device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader , val_dataset= data_loader()
    snn_dataset = SpikeDataset(val_dataset, T = args.T,theta = max_1-0.0001)
    #snn_dataset = SpikeDataset(val_dataset, T = args.T)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=10, shuffle=False)

    from models.vgg_imagenet_0_8 import CatVGG,VGG1
    #model1 = models.vgg19_bn(pretrained=True)
    #torch.save(model1.state_dict(), "imagevgg19bn_o.pt")
    #for param_tensor in model1.state_dict():
    #    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    model1 = VGG1('VGG16',bias = True).to(device)
    
    snn_model = CatVGG('VGG16', args.T, bias = True).to(device)
    #imageNmybn19_c_d_2_new_4_1218
    #imageNmybn19_c_d_2_new_4_1227_200_full
    model1.load_state_dict(torch.load("imageNet_t_32_40.pt"), strict=False)
    #for param_tensor in snn_model.state_dict():
    #    print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())
    #for param_tensor in model1.state_dict():
    #    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    #correct = test_(model1, device, val_loader)

    #torch.save(model1.state_dict(), "imageNmybn19_c_d_2_new.pt")
    correct_ = 0
    optimizer = optim.SGD(model1.parameters(), lr=args.lr,momentum = 0.9,weight_decay= 1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    k = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model1, device, train_loader, optimizer, epoch)
        correct = test(model1, device, val_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model1.state_dict(), "imageNmybn16_NIPS_t32.pt")
        k+=1
        scheduler.step()
    #torch.save(model1.state_dict(), "imageNet_t_32_40.pt")
    #torch.save(model1.state_dict(), "imageNmybn19_c_d_2_new_4_1218_.pt")

    #correct = test_(model1, device, val_loader)

    #model1 = fuse_bn_recursively(model)
    #torch.save(model.state_dict(), "imageNmybn19_c_d_2_fused.pt")
    model1 = fuse_bn_recursively(model1)
    #correct = test_(model1, device, val_loader)

    transfer_model(model1, snn_model)
    print("successful transfer")
    test_(snn_model, device, snn_loader)

    #test(snn_model, device, snn_loader)
    #if args.save_model
    #torch.save(model.state_dict(), "YOUR MOERL HERE.pt")
    
    
if __name__ == '__main__':
    main()

