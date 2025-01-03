from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, SpikeDataset ,load_model, fuse_module
import catCuda
import numpy as np

T = 4
timestep = 4
T_win = 1
T_reduce = 1
min_1 = 0
max_1 = T_win/T

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=T):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # constant
        #print("grad", grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

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
                #reshape_dict[k]=torch.cat((reshape_dict[k],reshape_dict[k]),-1)
                #print(reshape_dict[k].shape)
            else:
                reshape_dict[k] = nn.Parameter(v.reshape(dst_dict[k].shape))

    dst.load_state_dict(reshape_dict, strict=False)

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    #print("spikes_data",spikes_data.shape)
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    #print("out",out.shape)
    out = catCuda.getSpikes(out, max_1-0.0001)
    #print("out",out.shape)
    return out

class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
        #x = torch.clamp(x, min=min_1, max=max_1)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, T)), T),min=min_1, max=max_1)
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1: 

        self.conv1 = nn.Conv2d(1, 32, 3, 1,0, bias=True)
        self.Bn1 = nn.BatchNorm2d(32)
        self.conv1_ = nn.Conv2d(6, 32, 3, 1,0, bias=True)
        self.Bn1_ = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1,0, bias=True)
        self.Bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(32, 32, 4, 2,1, bias=True)
        self.Bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, 1,0, bias=True)
        self.Bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, 1,0, bias=True)
        self.Bn5 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.4)
        self.conv6 = nn.Conv2d(64, 64, 4, 2,1, bias=True)
        self.Bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, 3, 1,0, bias=True)
        self.Bn7 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4, 10, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)
        #self.fc2 = nn.Linear(128, 10, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        x = self.conv2(x)
        x = self.Bn2(x)

        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        x = self.conv3(x)
        x = self.Bn3(x)

        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)
        #x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.Bn4(x)
        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        x = self.conv5(x)
        x = self.Bn5(x)

        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        #x = self.dropout3(x)

        x = self.conv6(x)
        x = self.Bn6(x)

        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        #x = self.dropout2(x)

        x = self.conv7(x)
        x = self.Bn7(x)
        x7 = torch.max(x)

        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)
        #print(float(x1/1),float(x2/x1),float(x3/x2),float(x4/x3),float(x5/x4),float(x6/x5),float(x7/x6))
        
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        #x = torch.clamp(x, min=0, max=1)
        #x = torch.div(torch.floor(torch.mul(x,5)),5)
        #x = self.dropout3(x)

        #output = self.fc2(x)
        #output= torch.clamp(output, min=0, max=1)
        #x = torch.div(torch.ceil(torch.mul(x,50)),50)

        return x


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn

        self.conv1 = snn.conv(1, 32, 3, 1,0,bias=True)
        self.conv2 = snn.conv(32, 32, 3, 1,0,bias=True)
        self.conv3 = snn.conv(32, 32, 4,2,1,bias=True)

        self.conv4 = snn.conv(32, 64, 3, 1,0,bias=True)
        self.conv5 = snn.conv(64, 64, 3, 1,0,bias=True)
        self.conv6 = snn.conv(64, 64, 4, 2,1,bias=True)

        self.conv7 = snn.conv(64, 128, 3, 1,0,bias=True)
     
        self.fc1 = snn.dense((2,2,128), 10, bias=True)
        #self.fc2 = snn.dense(128, 10, bias=True)


    def forward(self, x):
    
        """
        y = self.conv1(x)
        #print(y.shape)
        #print(self.conv1.weight.shape)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        y = self.conv2(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)
        
        y = self.conv3(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)
        #print(torch.sum(y-z))
        y = self.conv4(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        y = self.conv5(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        y = self.conv6(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        y = self.conv7(y)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)
        #print(torch.sum(y))

        x = self.fc1(y)
        """
        x = self.snn.spike(self.conv1(x), theta = max_1-0.001)
        y_ = self.snn.spike(self.conv2(x), theta = max_1-0.001)
        y = self.snn.spike(self.conv3(y_), theta = max_1-0.001)
        x = self.snn.spike(self.conv4(y), theta = max_1-0.001)
        x = self.snn.spike(self.conv5(x), theta = max_1-0.001)
        x = self.snn.spike(self.conv6(x), theta =max_1-0.001)
        x = self.snn.spike(self.conv7(x), theta = max_1-0.001)
        x = self.fc1(x)
        
        return self.snn.sum_spikes(x)/self.T

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, onehot.type(torch.float))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    parser.add_argument('--T', type=int, default=T_win, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01),
        AddQuantization()
        ])

    transform=transforms.Compose([
        transforms.ToTensor(),
        AddQuantization()
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train)

    for i in range(30):
        transform_train_1 = transforms.Compose([
            transforms.RandomRotation(10),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            AddQuantization()
        ])
        dataset1 = dataset1+ datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train_1)


    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    snn_dataset = SpikeDataset(dataset2, T = args.T,theta = max_1-0.01)
    #print(type(dataset1[0][0]))
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    #print(test_loader[0])
    snn_loader = torch.utils.data.DataLoader(snn_dataset, **kwargs)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)
    model.load_state_dict(torch.load("MNSIT_t_1_4.pt"), strict=False)
    #model.load_state_dict(torch.load("Nips_MNIST_t1.pt"), strict=False)


    if args.resume != None:
        #load_model(torch.load(args.resume), model)
        model.load_state_dict(torch.load(args.resume), strict=False)

    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    ACC = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        ACC_ = test(model, device, test_loader)

        if ACC_>ACC or ACC_ == ACC:
            ACC = ACC_
            torch.save(model.state_dict(), "Nips_MNIST_t3_2_.pt")
            #fuse_module(model)
            #transfer_model(model, snn_model)
            #test(snn_model, device, snn_loader)
        
        scheduler.step()
    test(model, device, test_loader)
    fuse_module(model)
    #test(model, device, test_loader)
    transfer_model(model, snn_model)
    print("SNN")
    test(snn_model, device, snn_loader)
    #if args.save_model:



if __name__ == '__main__':
    main()
