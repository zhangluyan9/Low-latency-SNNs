from __future__ import print_function
import os
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, SpikeDataset
import catSNN
import catCuda
import numpy as np

T_reduce = 8
timestep = 10
timestep_f = 10
#VGG_5_
file_load_n = "../../pretrain_weight/cifar100/cifar100_NIPS_t8_10.pt"
#file_load = "cifar100_vggo_1109_60_full_09.pt"
#f_name = 'neuron_100_trysoa.npz'
min_1 = 0
max_1 = T_reduce/timestep
#max_1 = 1
num_ones_all = 0
cfg = {
    'o' : [128,128,'M',256,256,'M',512,512,'M',(1024,0),'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'o_low' : [(128,1,6.5162),(128,1,1.0253),'M',(256,1,1.6196),(256,1,0.8609),'M',(512,1,1.8470),(512,1,0.7465),'M',(1024,0,1.7244),'M'],
    'o_low_' : [(128,1,3.3402),(128,1,1.3331),'M',(256,1,1.1860),(256,1,0.8403),'M',(512,1,1.8164),(512,1,0.7146),'M',(1024,0,1.1968),'M'],


}
class NewSpike(nn.Module):
    def __init__(self, T=4):
        super(NewSpike, self).__init__()
        self.T = T
        self.total_num_ones = 0  # 新增的累积变量

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        num_ones = torch.sum(x).item()
        #print(num_ones)
        self.total_num_ones += num_ones  # 更新累积变量
        return x

    def get_total_num_ones(self):
        return self.total_num_ones

    def reset_total_num_ones(self):
        self.total_num_ones = 0  # 重置方法
    

class CatVGG(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier1 = self.snn.dense((1, 1, 1024), 100,bias = True)

    def forward(self, x):
        #print(torch.sum(x).item())
        out = self.features(x)
        out = self.classifier1(out)
        out_ = torch.sum(out)
        #print(float(out_))
        out = self.snn.sum_spikes(out) / self.T
        return out, out_

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 18
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2),NewSpike(self.T)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(T_reduce/timestep-0.001),nn.Identity()]
                    in_channels = x
                else:
                    padding = x[1] if isinstance(x, tuple) else 1
                    out_channels = x[0] if isinstance(x, tuple) else x
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               NewSpike(self.T ),nn.Identity()]
                    in_channels = out_channels
        return nn.Sequential(*layers)
    
    def get_total_num_ones(self):
        total = 0
        for layer in self.features:
            if isinstance(layer, NewSpike):
                total += layer.get_total_num_ones()
        return total

    def reset_total_num_ones(self):
        for layer in self.features:
            if isinstance(layer, NewSpike):
                layer.reset_total_num_ones()
    
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
        for i in range(1,T_reduce+1):
            x_origin_plus_1 = torch.where(x_origin_plus_1 == i/ (  (timestep + 1)), i*my_ones  / timestep, x_origin_plus_1)

        x_origin_minus_1 = torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep-1)), timestep-1),min=min_1, max=T_reduce/(timestep-1))
        my_ones = torch.ones(x_origin_minus_1.shape[0],x_origin_minus_1.shape[1],x_origin_minus_1.shape[2])
        for i in range(1,T_reduce+1):
            x_origin_minus_1 = torch.where(x_origin_minus_1 == i / ((timestep - 1)), i*my_ones   / timestep, x_origin_minus_1)
    
        x = torch.cat((x_origin, x_origin_plus_1,x_origin_minus_1,x_origin,x_origin_plus_1,x_origin_minus_1), 0)
        x_flattened = x.flatten()

        # 找出所有唯一的值
        #unique_values = set(x_flattened.tolist())
        #print(unique_values)

        return x

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


#timestep = 16
#timestep_f = 16.0
f_store = 'cifar_100_t_20.npz'

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    # t=1，2: T_reduce/timestep-0.001; 
    out = catCuda.getSpikes(out, T_reduce/timestep-0.001)
    return out
def spike_channel(input,Threshold):
    input = input.transpose(-4, -5)
    output = input.clone().cuda()
    for i in range(input.shape[0]):
        output[i] = catCuda.getSpikes(input[i].clone(), Threshold[i])

    output = output.transpose(-4, -5)
    return output



class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)

def fuse_bn_sequential(block):
    if not isinstance(block, nn.Sequential):
        return block
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']

                if 'weight' in bn_st_dict:
                    gamma = bn_st_dict['weight']
                else:
                    gamma = torch.ones(mu.size(0)).float().to(mu.device)
                #gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)

                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)

                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model



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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print(pred.eq(target.view_as(pred)).sum().item())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def test_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    s = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            output,out_ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            s +=out_

            #print(pred.eq(target.view_as(pred)).sum().item())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct,s



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=T_reduce, metavar='N',
                        help='SNN time window')
    parser.add_argument('--k', type=int, default=0, metavar='N',
                        help='Data augmentation')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                      )
    mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #AddGaussianNoise(std=0.01),
        AddQuantization_new()
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        AddQuantization_new()
    ])


    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

    for i in range(args.k):
        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization_new()
        ])
        trainset = trainset + datasets.CIFAR100(root='./data', train=True, download=True, transform=im_aug)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


    snn_dataset = SpikeDataset(testset, T = args.T,theta = max_1-0.001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=1000, shuffle=False)

    from models.vgg_vr_t1 import VGG_5,VGG_5_
    model= VGG_5_('o', clamp_max=1,bias =True).to(device)
    model.load_state_dict(torch.load(file_load_n), strict=False)

    snn_model = CatVGG('o', args.T, is_noise=False, bias = True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    correct_ = 0
    test(model, device, test_loader)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader_)
        correct = test(model, device, test_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model.state_dict(),"../../pretrain_weight/cifar100/cifar100_NIPS_t1_2dot5_tnnls.pt")
        scheduler.step()
    
    model = fuse_bn_recursively(model)

    transfer_model(model, snn_model)

    corr,add = test_(snn_model, device, snn_loader)
    #print(num_ones_all)
    total_num_ones = snn_model.get_total_num_ones()
    print("Total number of ones in the dataset:", total_num_ones/10000)
    #print(float(add/10000))
    #print("Total number of ones in the dataset:", total_num_ones)


if __name__ == '__main__':
    main()
