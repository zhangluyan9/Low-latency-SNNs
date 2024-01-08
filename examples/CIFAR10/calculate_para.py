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
from catSNN import  SpikeDataset ,fuse_bn_recursively
from models.vgg_0_5_0309 import VGG_o_,CatVGG_o
import catSNN
import catCuda
T_reduce = 8
timestep = 10
timestep_f = 10
min_1 = 0
max_1 = T_reduce/timestep
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'o' : [(128,1,1),(128,1,2),'M',(256,1,3),(256,1,4),'M',(512,1,5),(512,1,6),'M',(1024,0,7),'M'],
    'o_low' : [(128,1,6.8658),(128,1,0.8518),'M',(256,1,1.5976),(256,1,0.8886),'M',(512,1,1.7140),(512,1,0.6957),'M',(1024,0,1.7274),'M'],

}
class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=max_1,q_level = timestep_f):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max
        self.q_level = q_level

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = Quantization_(x, self.q_level)
        return x
class NewSpike(nn.Module):
    def __init__(self, T = T_reduce):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):

        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x

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
        #0/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10

        x_origin_plus_1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep+1)), timestep+1),min=min_1, max=T_reduce/(timestep+1))
        #0/11,1/11,2/11,3/11,4/11,5/11,6/11,7/11,8/11

        my_ones = torch.ones(x_origin_plus_1.shape[0],x_origin_plus_1.shape[1],x_origin_plus_1.shape[2])
        for i in range(0,T_reduce+1):
            x_origin_plus_1 = torch.where(x_origin_plus_1 == i/(timestep + 1), i*my_ones/timestep, x_origin_plus_1)

        x_origin_minus_1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep-1)), timestep-1),min=min_1, max=T_reduce/(timestep-1))
        my_ones = torch.ones(x_origin_minus_1.shape[0],x_origin_minus_1.shape[1],x_origin_minus_1.shape[2])
        for i in range(0,T_reduce+1):
            x_origin_minus_1 = torch.where(x_origin_minus_1 == i / ((timestep - 1)), i*my_ones   / timestep, x_origin_minus_1)
    
        x = torch.cat((x_origin_minus_1, x_origin_plus_1,x_origin ,x_origin_minus_1,x_origin_plus_1,x_origin), 0)
        #x = torch.cat((x_origin, x_origin_plus_1,x_origin_minus_1,x_origin,x_origin_plus_1,x_origin_minus_1), 0)

        # 找出所有唯一的值
        #unique_values = set(x_flattened.tolist())
        #print(unique_values)

        return x

class AddQuantization_new_(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
        #x = torch.clamp(x, min=min_1, max=max_1)
        x1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)

        x2 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep+1)), timestep+1),min=min_1, max=T_reduce/(timestep+1))
        my_ones = torch.ones(x2.shape[0],x2.shape[1],x2.shape[2])
        x2 = torch.where(x2==T_reduce/(timestep+1) , my_ones*max_1, x2)



        x3 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep-1)), timestep-1),min=min_1, max=T_reduce/(timestep-1))
        my_ones = torch.ones(x3.shape[0],x3.shape[1],x3.shape[2])
        x3 = torch.where(x3==T_reduce/(timestep-1) , my_ones*max_1, x3)
        
        x = torch.cat((x3, x2,x1,x3,x2,x1), 0)
        #print(x)
        #print(x.shape)
        return x


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

class VGG_o_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_o_, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier4 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        #print(self.features[0](x)).size
        out = self.features(x)
        #print((out))
        out = out.view(out.size(0), -1)
        out = self.classifier4(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        #turn to 18 it T = 1; ALSO, THE NAME OF LAST LAYER IS "classifier4" insted of "classifier1", PLEASE CHANGE IF YOU USE OUR PRETRAIN MODEL
        in_channels = 3

        for x in cfg:
            if x == 'M':
                #layers += [nn.AvgPool2d(kernel_size=2, stride=2),Act_op()]
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_()]

            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                #Act_op()
                #layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=self.bias),nn.BatchNorm2d(out_channels),Act_op(),nn.Dropout2d(0.1)]
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=self.bias),nn.BatchNorm2d(out_channels),Clamp_q_(),nn.Dropout2d(0.1)]
                in_channels = out_channels

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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
    parser.add_argument('--data_augment', type=str, default=False, metavar='N',
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
    #data_augment = args.data_augment
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        #turn to AddQuantization_new_ if T = 1
        AddQuantization()
        #AddQuantization_new()
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization(),
        #AddQuantization_new()
        ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

    
    train_loader = torch.utils.data.DataLoader(
       trainset, batch_size=256+512, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=15*8, shuffle=False)


    snn_dataset = SpikeDataset(testset, T = args.T,theta = max_1-0.001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=500, shuffle=False)

    #model = VGG_o_('o', clamp_max=1,bias =True).to(device)

    model = VGG_o_('VGG16', clamp_max=1,bias =True).to(device)
    
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 打印总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 打印总参数量
    print("Total number of trainable parameters in VGG_o_ model is:", total_params)
if __name__ == '__main__':
    main()
