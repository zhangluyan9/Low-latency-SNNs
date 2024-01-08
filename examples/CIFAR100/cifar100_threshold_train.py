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

T_reduce = 16
timestep = 20
timestep_f = 20
file_load = "../../pretrain_weight/cifar100/cifar100_NIPS_t16_20.pt"
#cifar100_vggo_1109_10_16_cTT
#cifar100_NIPS_t8_10
weight_load = '../../pretrain_weight/cifar100/cifar_100_t_16_20_.npz'
#f_name = 'neuron_100_trysoa.npz'
min_1 = 0
max_1 = T_reduce/timestep
f_store = '../../pretrain_weight/cifar100/cifar_100_t_16_20_tnnls1.npz'
#max_1 = 1


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

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, T_reduce/timestep-0.001)
    return out
class NewSpike(nn.Module):
    def __init__(self, T = T_reduce):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x

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

def spike_channel(input,Threshold):
    input = input.transpose(-4, -5)
    output = input.clone().cuda()
    for i in range(input.shape[0]):
        output[i] = catCuda.getSpikes(input[i].clone(), Threshold[i])
    output = output.transpose(-4, -5)
    return output

def threshold_training_snn(input, y_origin,threshold,T,index):
    #print(index)
    #print(threshold[0])
    threshold_pre_1 = threshold
    mul =  input.shape[0] *input.shape[2] *input.shape[3] *input.shape[4]  
    y_new = spike_channel(input.clone(),threshold_pre_1)
    y = (torch.sum(y_origin.clone(), dim=4)) /T
    y = create_spike_input_cuda(y, T)
    
    threshold_1 =  threshold_pre_1
    
    for i in range (y.shape[1]):
        j = 0 
        diff = (torch.sum(y.transpose(-4, -5)[i])-torch.sum(y_new.transpose(-4, -5)[i]))/mul
        #print("before",diff)
        diff_ = torch.sum(torch.logical_xor(y.transpose(-4, -5)[i],y_new.transpose(-4, -5)[i]))/mul
        threshold_1[i] = threshold_1[i] - 0.1*diff*max_1
        """
        if  diff > 5e-2 or diff < - 5e-2: 
            threshold_1[i] = threshold_1[i] - diff
        elif diff > 5e-3 or diff < - 5e-3:
            threshold_1[i] = threshold_1[i] - 0.1*diff
        elif diff > 5e-4 or diff < - 5e-4:
            threshold_1[i] = threshold_1[i] - 0.01*diff
        while diff_ > 4e-3 or diff_ < - 4e-3:
            j+=1
            threshold_1[i] = threshold_1[i] - 0.1*diff_*diff/torch.abs(diff)
            y_new = spike_channel((input).clone(),threshold_1)
            diff = (torch.sum(y.transpose(-4, -5)[i])-torch.sum(y_new.transpose(-4, -5)[i]))/mul
            diff_ = torch.sum(torch.logical_xor(y.transpose(-4, -5)[i],y_new.transpose(-4, -5)[i]))/mul
            #print(j)
            if j >=30:
                break
        """ 
        #print("after",diff)
    
    #print(threshold_pre_1[0])
    #print(threshold_1[0])

    threshold_pre_1 =  threshold_1 
    return y_new,y

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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(pred.eq(target.view_as(pred)).sum().item())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct

def test_snn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 78 77 77
    #9 9654 8 9697 7 9712 6 9734
    
    f = np.load(weight_load)
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
            np.savez(f_store,threshold_pre_1=threshold_pre_1,threshold_pre_2=threshold_pre_2,threshold_pre_3=threshold_pre_3,threshold_pre_4=threshold_pre_4,threshold_pre_5=threshold_pre_5,threshold_pre_6=threshold_pre_6,threshold_pre_7=threshold_pre_7,p1=p1,p2=p2,p3=p3,p4=p4)

    test_loss /= len(test_loader.dataset)
    #np.savez('cifar_30_lay_chann.npz',threshold_pre_1=threshold_pre_1,threshold_pre_2=threshold_pre_2,threshold_pre_3=threshold_pre_3,threshold_pre_4=threshold_pre_4,threshold_pre_5=threshold_pre_5,threshold_pre_6=threshold_pre_6,threshold_pre_7=threshold_pre_7,p1=p1,p2=p2,p3=p3,p4=p4)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    

    return correct
#'o' : [(128,1,1),(128,1,2),'M',(256,1,3),(256,1,4),'M',(512,1,5),(512,1,6),'M',(1024,0,7),'M'],
#self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),self.snn.spikeLayer(1.0)

class CatNet_training(nn.Module):

    def __init__(self, T):
        super(CatNet_training, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn
        self.conv1 = snn.conv(18, 128, kernelSize=3, padding=1,bias=True)
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
        
        self.classifier1 = snn.dense((1,1,1024), 100, bias=True)
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
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
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
    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=512+256, shuffle=True)

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

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256+512, shuffle=True)

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)


    snn_dataset = SpikeDataset(trainset, T = args.T,theta = (max_1-0.00001))
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=100, shuffle=False)

    from models.vgg_vr_t1 import VGG_5,CatVGG,VGG_5_
    model = VGG_5_('o', clamp_max=1.0, bias = True).to(device)
    #cifar100_vggo_0505_2
    #vggo_largeinput_3
    #cifar100_vggo_16_1109_
    #cifar100_vggo_16_1109
    
    model.load_state_dict(torch.load(file_load), strict=False)
    snn_model = CatVGG('o', args.T, is_noise=False, bias = True).to(device)
    snn_model_training = CatNet_training(args.T).to(device)

    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())
    if args.resume != None:
        model.load_state_dict(torch.load(args.resume))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    correct_ = 0
    #test(model, device, test_loader)
    #torch.save(model.state_dict(), "cifar100_t_24_30.pt")

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader_)
        correct = test(model, device, test_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model.state_dict(), "cifar100_vggo_1109_32_40_cTT.pt")
        scheduler.step()
    
    model = fuse_bn_recursively(model)
    transfer_model(model, snn_model)
    #test_(snn_model, device, snn_loader)
    
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

    reshape_dict['pool1.weight'] = nn.Parameter(pre_dict['features.6.weight'],requires_grad=False)
    reshape_dict['pool2.weight'] = nn.Parameter(pre_dict['features.14.weight'],requires_grad=False)
    reshape_dict['pool3.weight'] = nn.Parameter(pre_dict['features.22.weight'],requires_grad=False)
    reshape_dict['pool4.weight'] = nn.Parameter(pre_dict['features.27.weight'],requires_grad=False)

    reshape_dict['classifier1.weight']=nn.Parameter(pre_dict['classifier1.weight'],requires_grad=False)
    reshape_dict['classifier1.bias']=nn.Parameter(pre_dict['classifier1.bias'],requires_grad=False)

    model_dict.update(reshape_dict)
    snn_model_training.load_state_dict(model_dict)

    #with torch.no_grad():
    #    normalize_weight(snn_model.features, quantize_bit=11)
    #test_(snn_model, device, snn_loader)
    print("start")
    test_snn(snn_model_training, device, snn_loader)
    #test_(snn_model, device, snn_loader)
    
    #model = fuse_bn_recursively(model)
    #transfer_model(model, snn_model)
    #test_(snn_model, device, snn_loader)
    
    


if __name__ == '__main__':
    main()
