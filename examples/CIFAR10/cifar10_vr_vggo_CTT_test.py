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
import time
from catSNN import spikeLayer,  SpikeDataset ,fuse_bn_recursively
from models.vgg_0_5_0309 import VGG_o_,CatVGG_o
import catSNN
import catCuda

T_reduce = 32
timestep = 40
timestep_f = 40.0

min_1 = 0
max_1 = T_reduce/timestep
#file_load = "cifar100_vggo_0505_2.pt"
f_store  = 'cifar_10_t_32_40.npz'


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
    reshape_dict['features.22.weight'] = dst_dict['features.22.weight']*max_1
    reshape_dict['features.27.weight'] = dst_dict['features.27.weight']*max_1
    dst.load_state_dict(reshape_dict, strict=False)


class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, T_reduce/timestep-0.001)
    return out

def change_shape(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

def spike_channel(input,Threshold):
    input = input.transpose(-4, -5)
    output = input.clone().cuda()
    for i in range(input.shape[0]):
        output[i] = catCuda.getSpikes(input[i].clone(), Threshold[i])
    output = output.transpose(-4, -5)
    return output


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
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct



def threshold_training_snn(input, y_origin,threshold,T,index):
    #print(index)
    #print(threshold[0])
    threshold_pre_1 = threshold
    mul =  input.shape[0] *input.shape[2] *input.shape[3] *input.shape[4]  
    y_new = spike_channel(input.clone(),threshold_pre_1)
    y = (torch.sum(y_origin.clone(), dim=4)) /T
    y = create_spike_input_cuda(y, T)
    return y_new,y



def test_snn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 78 77 77
    #9 9654 8 9697 7 9712 6 9734
    
    f = np.load(f_store)
    threshold_pre_1 = f['threshold_pre_1']
    threshold_pre_2 = f['threshold_pre_2']
    threshold_pre_3 = f['threshold_pre_3']
    threshold_pre_4 = f['threshold_pre_4']
    threshold_pre_5 = f['threshold_pre_5']
    threshold_pre_6 = f['threshold_pre_6']
    threshold_pre_7 = f['threshold_pre_7']

    p1 = f['p1']
    p2 = f['p2']
    p3 = f['p3']
    p4 = f['p4']

    with torch.no_grad():
        i = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4 = model(data,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4)
            test_loss +=  F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(pred.eq(target.view_as(pred)).sum().item())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    

    return correct


class CatNet_training(nn.Module):

    def __init__(self, T):
        super(CatNet_training, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn
        self.conv1 = snn.conv(3, 128, kernelSize=3, padding=1,bias=True)
        self.conv2 = snn.conv(128, 128, kernelSize=3, padding=1,bias=True)
        self.pool1 = snn.pool(2)
        self.conv3= snn.conv(128, 256, kernelSize=3, padding=1,bias=True)
        self.conv4= snn.conv(256, 256, kernelSize=3, padding=1,bias=True)
        self.pool2 = snn.pool(2)
        self.conv5= snn.conv(256, 512, kernelSize=3, padding=1,bias=True)
        self.conv6 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.pool3 = snn.pool(2)
        self.conv7 = snn.conv(512, 1024, kernelSize=3, padding=0,bias=True)
        self.pool4 = snn.pool(2)
        
        self.classifier1 = snn.dense((1,1,1024), 10, bias=True)
        #self.fc2 = snn.dense(128, 10, bias=True)


    def forward(self, x,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4):
    #def forward(self, x):
        #x,indices = torch.sort(x, descending=True) 
        
        x,y = threshold_training_snn(self.conv1(x),self.conv1(x),threshold_pre_1,self.T,1)
        x,y = threshold_training_snn(self.conv2(x),self.conv2(y),threshold_pre_2,self.T,2)
        x,y = threshold_training_snn(self.pool1(x),self.pool1(y),p1,self.T,3)
        x,y = threshold_training_snn(self.conv3(x),self.conv3(y),threshold_pre_3,self.T,4)
        x,y = threshold_training_snn(self.conv4(x),self.conv4(y),threshold_pre_4,self.T,5)
        x,y = threshold_training_snn(self.pool2(x),self.pool2(y),p2,self.T,6)
        x,y = threshold_training_snn(self.conv5(x),self.conv5(y),threshold_pre_5,self.T,7)
        x,y = threshold_training_snn(self.conv6(x),self.conv6(y),threshold_pre_6,self.T,8)
        x,y = threshold_training_snn(self.pool3(x),self.pool3(y),p3,self.T,9)
        x,y = threshold_training_snn(self.conv7(x),self.conv7(y),threshold_pre_7,self.T,10)
        x,y = threshold_training_snn(self.pool4(x),self.pool4(y),p4,self.T,11)
        """
        fac =  T_reduce/timestep
        x = self.snn.spike(self.conv1(x),fac)
        x = self.snn.spike(self.conv2(x),fac)
        x = self.snn.spike(self.pool1(x),fac)
        x = self.snn.spike(self.conv3(x),fac)
        x = self.snn.spike(self.conv4(x),fac)
        x = self.snn.spike(self.pool2(x),fac)
        x = self.snn.spike(self.conv5(x),fac)
        x = self.snn.spike(self.conv6(x),fac)
        x = self.snn.spike(self.pool3(x),fac)
        x = self.snn.spike(self.conv7(x),fac)
        x = self.snn.spike(self.pool4(x),fac)
        """
        x = self.classifier1(x)
        return self.snn.sum_spikes(x)/self.T,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,p1,p2,p3,p4



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
        
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #AddGaussianNoise(std=0.01),
        AddQuantization()
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
       
    for i in range(args.k):

        im_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR10(root='./data', train=True, download=True, transform=im_aug)

    for i in range(args.k):
        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])
        trainset = trainset + datasets.CIFAR10(root='./data', train=True, download=True, transform=im_aug)
    
    train_loader = torch.utils.data.DataLoader(
       trainset, batch_size=512+256, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=15*8, shuffle=False)


    snn_dataset = SpikeDataset(testset, T = args.T,theta = max_1-0.001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=100, shuffle=False)

    
    model = VGG_o_('o', clamp_max=1,bias =True).to(device)
    #t_training_20_

    if args.resume != None:
        #load_model(torch.load(args.resume), model)
        model.load_state_dict(torch.load(args.resume), strict=False)

    snn_model = CatVGG_o('o', args.T,bias =True).to(device)
    snn_model_training = CatNet_training(args.T).to(device)

    

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
            torch.save(model.state_dict(), "cifar10_0_5_ASG_70_08_.pt")

        scheduler.step()
    
    model = fuse_bn_recursively(model)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())

    transfer_model(model, snn_model)
    #test(snn_model, device, snn_loader)
    #with torch.no_grad():
    #    normalize_weight(snn_model.features, quantize_bit=8)

    
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
    

if __name__ == '__main__':
    main()
