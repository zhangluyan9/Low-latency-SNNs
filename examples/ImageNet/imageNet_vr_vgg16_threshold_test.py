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
import catSNN
import catCuda
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.models as models
weight_load = "image_200_full_1.npz"
T_reduce = 200
timestep = 250
timestep_f = 250
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
    
    return train_loader, val_loader, val_dataset,train_dataset

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

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    #print("spikes_data",spikes_data.shape)
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    #print("out",out.shape)
    out = catCuda.getSpikes(out, max_1 - 0.0001)
    #print("out",out.shape)
    return out

def spike_channel(input,Threshold):
    input = input.transpose(-4, -5)
    output = input.clone().cuda()
    for i in range(input.shape[0]):
        output[i] = catCuda.getSpikes(input[i].clone(), Threshold[i])

    output = output.transpose(-4, -5)
    return output  

def threshold_training_snn(input,threshold):
    #print(index)
    #print(threshold[0])
    #print(input.shape)
    threshold_pre_1 = threshold
    y_new = spike_channel(input.clone(),threshold_pre_1)
    return y_new

class CatNet_training(nn.Module):

    def __init__(self, T):
        super(CatNet_training, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn
        #    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        #    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

        self.conv1 = snn.conv(3, 64, kernelSize=3, padding=1,bias=True)
        self.conv2 = snn.conv(64, 64, kernelSize=3, padding=1,bias=True)
        self.pool1 = snn.pool(2)
        self.conv3 = snn.conv(64, 128, kernelSize=3, padding=1,bias=True)
        self.conv4 = snn.conv(128, 128, kernelSize=3, padding=1,bias=True)
        self.pool2 = snn.pool(2)
        self.conv5 = snn.conv(128, 256, kernelSize=3, padding=1,bias=True)
        self.conv6 = snn.conv(256, 256, kernelSize=3, padding=1,bias=True)
        self.conv7 = snn.conv(256, 256, kernelSize=3, padding=1,bias=True)
        self.pool3 = snn.pool(2)
        self.conv8 = snn.conv(256, 512, kernelSize=3, padding=1,bias=True)
        self.conv9 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.conv10 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.pool4 = snn.pool(2)
        self.conv11 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.conv12 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.conv13 = snn.conv(512, 512, kernelSize=3, padding=1,bias=True)
        self.pool5 = snn.pool(2)
        self.classifier0 = self.snn.dense((7,7,512),4096, bias=True)
        self.classifier3 = self.snn.dense((1,1,4096),4096, bias=True)
        self.classifier6 = self.snn.dense((1,1,4096),1000, bias=True)

        #self.fc2 = snn.dense(128, 10, bias=True)


    def forward(self, x,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,threshold_pre_8,threshold_pre_9,threshold_pre_10,threshold_pre_11,threshold_pre_12,threshold_pre_13,threshold_pre_14,threshold_pre_15,p1,p2,p3,p4,p5):
    #def forward(self, x):
        x = threshold_training_snn(self.conv1(x),threshold_pre_1)
        x = threshold_training_snn(self.conv2(x),threshold_pre_2)
        x = threshold_training_snn(self.pool1(x),p1)

        x = threshold_training_snn(self.conv3(x),threshold_pre_3)
        x = threshold_training_snn(self.conv4(x),threshold_pre_4)
        x = threshold_training_snn(self.pool2(x),p2)

        x = threshold_training_snn(self.conv5(x),threshold_pre_5)
        x = threshold_training_snn(self.conv6(x),threshold_pre_6)
        x = threshold_training_snn(self.conv7(x),threshold_pre_7)
        x = threshold_training_snn(self.pool3(x),p3)

        x = threshold_training_snn(self.conv8(x),threshold_pre_8)
        x = threshold_training_snn(self.conv9(x),threshold_pre_9)
        x = threshold_training_snn(self.conv10(x),threshold_pre_10)

        x = threshold_training_snn(self.pool4(x),p4)

        x = threshold_training_snn(self.conv11(x),threshold_pre_8)
        x = threshold_training_snn(self.conv12(x),threshold_pre_9)
        x = threshold_training_snn(self.conv13(x),threshold_pre_10)
        x = threshold_training_snn(self.pool5(x),p5)

        x = threshold_training_snn(self.classifier0(x),threshold_pre_14)
        x = threshold_training_snn(self.classifier3(x),threshold_pre_15)
        x = self.classifier6(x)
        return self.snn.sum_spikes(x)/self.T,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,threshold_pre_8,threshold_pre_9,threshold_pre_10,threshold_pre_11,threshold_pre_12,threshold_pre_13,threshold_pre_14,threshold_pre_15,p1,p2,p3,p4,p5

def test_snn(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 78 77 77
    #9 9654 8 9697 7 9712 6 9734
    #    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        #    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    f = np.load(weight_load)
    threshold_pre_1 = f['threshold_pre_1']
    threshold_pre_2 = f['threshold_pre_2']
    threshold_pre_3 = f['threshold_pre_3']
    threshold_pre_4 = f['threshold_pre_4']
    threshold_pre_5 = f['threshold_pre_5']
    threshold_pre_6 = f['threshold_pre_6']
    threshold_pre_7 = f['threshold_pre_7']
    threshold_pre_8 = f['threshold_pre_8']
    threshold_pre_9 = f['threshold_pre_9']
    threshold_pre_10 = f['threshold_pre_10']
    threshold_pre_11 = f['threshold_pre_11']
    threshold_pre_12 = f['threshold_pre_12']
    threshold_pre_13 = f['threshold_pre_13']
    threshold_pre_14 = f['threshold_pre_14']
    threshold_pre_15 = f['threshold_pre_15']


    p1 = f['p1']
    p2 = f['p2']
    p3 = f['p3']
    p4 = f['p4']
    p5 = f['p5']
    

    with torch.no_grad():
        i = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,threshold_pre_8,threshold_pre_9,threshold_pre_10,threshold_pre_11,threshold_pre_12,threshold_pre_13,threshold_pre_14,threshold_pre_15,p1,p2,p3,p4,p5 = model(data,threshold_pre_1,threshold_pre_2,threshold_pre_3,threshold_pre_4,threshold_pre_5,threshold_pre_6,threshold_pre_7,threshold_pre_8,threshold_pre_9,threshold_pre_10,threshold_pre_11,threshold_pre_12,threshold_pre_13,threshold_pre_14,threshold_pre_15,p1,p2,p3,p4,p5)
            test_loss +=  F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(pred.eq(target.view_as(pred)).sum().item())
            #np.savez('image_200_full.npz',threshold_pre_1=threshold_pre_1,threshold_pre_2=threshold_pre_2,threshold_pre_3=threshold_pre_3,threshold_pre_4=threshold_pre_4,threshold_pre_5=threshold_pre_5,threshold_pre_6=threshold_pre_6,threshold_pre_7=threshold_pre_7,threshold_pre_8=threshold_pre_8,threshold_pre_9 = threshold_pre_9,threshold_pre_10 = threshold_pre_10,threshold_pre_11 = threshold_pre_11,threshold_pre_12=threshold_pre_12,threshold_pre_13=threshold_pre_13,threshold_pre_14=threshold_pre_14,threshold_pre_15=threshold_pre_15,p1=p1,p2=p2,p3=p3,p4=p4,p5=p5)

    test_loss /= len(test_loader.dataset)
    #np.savez('cifar_30_lay_chann.npz',threshold_pre_1=threshold_pre_1,threshold_pre_2=threshold_pre_2,threshold_pre_3=threshold_pre_3,threshold_pre_4=threshold_pre_4,threshold_pre_5=threshold_pre_5,threshold_pre_6=threshold_pre_6,threshold_pre_7=threshold_pre_7,p1=p1,p2=p2,p3=p3,p4=p4)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    
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

    train_loader, val_loader , val_dataset,train_dataset= data_loader()
    snn_dataset = SpikeDataset(val_dataset, T = args.T,theta = max_1-0.0001)
    #snn_dataset = SpikeDataset(val_dataset, T = args.T)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=2, shuffle=False)

    from models.vgg_imagenet_0_8 import CatVGG,VGG1
    #model1 = models.vgg19_bn(pretrained=True)
    #torch.save(model1.state_dict(), "imagevgg19bn_o.pt")
    #for param_tensor in model1.state_dict():
    #    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
    model1 = VGG1('VGG16',bias = True).to(device)
    
    snn_model = CatVGG('VGG16', args.T, bias = True).to(device)
    snn_model_training = CatNet_training(args.T).to(device)

    #imageNmybn19_c_d_2_new_4_1218
    #imageNmybn19_c_d_2_new_4_1227_200_full
    model1.load_state_dict(torch.load("imageNmybn16_NIPS_t140.pt"), strict=False)
    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())
    for param_tensor in snn_model_training.state_dict():
        print(param_tensor, "\t", snn_model_training.state_dict()[param_tensor].size())
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
            torch.save(model1.state_dict(), "imageNmybn16_NIPS_t200.pt")
        k+=1
        scheduler.step()
    #torch.save(model1.state_dict(), "imageNet_t_32_40.pt")
    #torch.save(model1.state_dict(), "imageNmybn19_c_d_2_new_4_1218_.pt")

    #correct = test_(model1, device, val_loader)

    #model1 = fuse_bn_recursively(model)
    #torch.save(model.state_dict(), "imageNmybn19_c_d_2_fused.pt")
    model1 = fuse_bn_recursively(model1)

    transfer_model(model1, snn_model)
    #correct = test_(model1, device, val_loader)

    print("successful transfer")
    #test_(snn_model, device, snn_loader)

    model_dict = snn_model_training.state_dict()
    pre_dict = snn_model.state_dict()

    reshape_dict = {}
    reshape_dict['conv1.weight'] = nn.Parameter(pre_dict['features.0.weight'],requires_grad=False)
    reshape_dict['conv1.bias'] = nn.Parameter(pre_dict['features.0.bias'],requires_grad=False)

    reshape_dict['conv2.weight'] = nn.Parameter(pre_dict['features.2.weight'],requires_grad=False)
    reshape_dict['conv2.bias'] = nn.Parameter(pre_dict['features.2.bias'],requires_grad=False)

    reshape_dict['conv3.weight'] = nn.Parameter(pre_dict['features.6.weight'],requires_grad=False)
    reshape_dict['conv3.bias'] = nn.Parameter(pre_dict['features.6.bias'],requires_grad=False)

    reshape_dict['conv4.weight'] = nn.Parameter(pre_dict['features.8.weight'],requires_grad=False)
    reshape_dict['conv4.bias'] = nn.Parameter(pre_dict['features.8.bias'],requires_grad=False)

    reshape_dict['conv5.weight'] = nn.Parameter(pre_dict['features.12.weight'],requires_grad=False)
    reshape_dict['conv5.bias'] = nn.Parameter(pre_dict['features.12.bias'],requires_grad=False)

    reshape_dict['conv6.weight'] = nn.Parameter(pre_dict['features.14.weight'],requires_grad=False)
    reshape_dict['conv6.bias'] = nn.Parameter(pre_dict['features.14.bias'],requires_grad=False)

    reshape_dict['conv7.weight'] = nn.Parameter(pre_dict['features.16.weight'],requires_grad=False)
    reshape_dict['conv7.bias'] = nn.Parameter(pre_dict['features.16.bias'],requires_grad=False)

    reshape_dict['conv8.weight'] = nn.Parameter(pre_dict['features.20.weight'],requires_grad=False)
    reshape_dict['conv8.bias'] = nn.Parameter(pre_dict['features.20.bias'],requires_grad=False)

    reshape_dict['conv9.weight'] = nn.Parameter(pre_dict['features.22.weight'],requires_grad=False)
    reshape_dict['conv9.bias'] = nn.Parameter(pre_dict['features.22.bias'],requires_grad=False)

    reshape_dict['conv10.weight'] = nn.Parameter(pre_dict['features.24.weight'],requires_grad=False)
    reshape_dict['conv10.bias'] = nn.Parameter(pre_dict['features.24.bias'],requires_grad=False)

    reshape_dict['conv11.weight'] = nn.Parameter(pre_dict['features.28.weight'],requires_grad=False)
    reshape_dict['conv11.bias'] = nn.Parameter(pre_dict['features.28.bias'],requires_grad=False)

    reshape_dict['conv12.weight'] = nn.Parameter(pre_dict['features.30.weight'],requires_grad=False)
    reshape_dict['conv12.bias'] = nn.Parameter(pre_dict['features.30.bias'],requires_grad=False)

    reshape_dict['conv13.weight'] = nn.Parameter(pre_dict['features.32.weight'],requires_grad=False)
    reshape_dict['conv13.bias'] = nn.Parameter(pre_dict['features.32.bias'],requires_grad=False)


    reshape_dict['pool1.weight'] = nn.Parameter(pre_dict['features.4.weight'],requires_grad=False)
    reshape_dict['pool2.weight'] = nn.Parameter(pre_dict['features.10.weight'],requires_grad=False)
    reshape_dict['pool3.weight'] = nn.Parameter(pre_dict['features.18.weight'],requires_grad=False)
    reshape_dict['pool4.weight'] = nn.Parameter(pre_dict['features.26.weight'],requires_grad=False)
    reshape_dict['pool5.weight'] = nn.Parameter(pre_dict['features.34.weight'],requires_grad=False)

    reshape_dict['classifier0.weight']=nn.Parameter(pre_dict['classifier0.weight'],requires_grad=False)
    reshape_dict['classifier0.bias']=nn.Parameter(pre_dict['classifier0.bias'],requires_grad=False)

    reshape_dict['classifier3.weight']=nn.Parameter(pre_dict['classifier3.weight'],requires_grad=False)
    reshape_dict['classifier3.bias']=nn.Parameter(pre_dict['classifier3.bias'],requires_grad=False)

    reshape_dict['classifier6.weight']=nn.Parameter(pre_dict['classifier6.weight'],requires_grad=False)
    reshape_dict['classifier6.bias']=nn.Parameter(pre_dict['classifier6.bias'],requires_grad=False)

    model_dict.update(reshape_dict)
    snn_model_training.load_state_dict(model_dict)
    #test_snn(snn_model_training, device, snn_loader)

    #test(snn_model, device, snn_loader)
    #if args.save_model
    #torch.save(model.state_dict(), "YOUR MOERL HERE.pt")
    
    
if __name__ == '__main__':
    main()

