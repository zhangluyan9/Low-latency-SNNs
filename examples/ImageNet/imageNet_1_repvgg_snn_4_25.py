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
import PIL

from torch.utils.data.sampler import SubsetRandomSampler
import catCuda,catSNN

import torchvision.models as models
max_1 = 4.0
timestep_f = 25
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M'],
    'repvgg_' : [(64,2),(192,2),192,192,192,(384,2),384,384,384,384,384,(768,2),768,768,768,768,768,768,768,768,768,768,768,768,768,768,768,(2560,2)],
}
def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, max_1-0.001)
    return out
class NewSpike(nn.Module):
    def __init__(self, T = 16):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x

class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=max_1):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = torch.div(torch.floor(torch.mul(x, timestep_f)), timestep_f)
        return x
        
class VGG(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG, self).__init__()
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier1 = nn.Linear(2560, 1000, bias=True)
        self.gap = nn.AvgPool2d(kernel_size=10)



    def forward(self, x):
        #with torch.no_grad():
        out = self.features(x)
        #print(out.shape)
        #out = self.features(x)
        out = self.gap(out)
        out = torch.clamp(out, min=0, max=2)
        out = torch.div(torch.floor(torch.mul(out,timestep_f)),timestep_f)

        #print(out.shape)

        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.classifier1(out)
        return out
 


    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if isinstance(x, tuple):
                layers += [nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1,stride = x[1], bias=self.bias,groups=1),Clamp_q_()]
                #if self.quantize_factor!=-1:
                #    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x[0]            
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias,groups=1),
                           Clamp_q_()]#catSNN.Clamp(max = self.clamp_max)
                #if self.quantize_factor!=-1:
                #    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class CatVGG(nn.Module):
    def __init__(self, vgg_name, T, bias=True):
        super(CatVGG, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias
        self.features = self._make_layers(cfg[vgg_name])
        #self.gap1 = self.snn.pool(7)
        self.classifier1 = self.snn.dense((1,1,2560),1000, bias=True)


    def forward(self, x):
        out = self.features(x)
        #y = (torch.sum(out, dim=4)) /self.T
        #out = create_spike_input_cuda(y, self.T)
        #out =  self.gap1(out)
        out = self.classifier1(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if isinstance(x, tuple):
                layers += [self.snn.conv(in_channels, x[0], kernelSize=3, padding=1,stride = x[1], bias=self.bias,groups=1),NewSpike(self.T)]
                #if self.quantize_factor!=-1:
                #    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x[0]            
            else:
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias,groups=1),
                           NewSpike(self.T)]#catSNN.Clamp(max = self.clamp_max)
                #if self.quantize_factor!=-1:
                #    layers += [catSNN.Quantize(self.quantize_factor)]
                in_channels = x
        # layers += [self.snn.sum_spikes_layer()]
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #,NewSpike(self.T)
        layers += [self.snn.pool(10),NewSpike(self.T) ]
        return nn.Sequential(*layers)

class AddQuantization(object):
    def __init__(self, min=0., max=max_1):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)

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
    reshape_dict['features.56.weight'] = dst_dict['features.56.weight'] *max_1
    dst.load_state_dict(reshape_dict, strict=False)

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
    """
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(320, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            AddQuantization()

        ])
    )
    """
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(320, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            AddQuantization()
            #normalize
        ])
    )
    #val_dataset_100 = Subset(val_dataset, range(0,15000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    num_training_samples = 10 
    train_sampler = SubsetRandomSampler(torch.arange(1000, 1100))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=250,
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
    #print("test_1")
    with torch.no_grad():
        for data, target in test_loader:
            #print("test_2")
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
    print("test_1")
    with torch.no_grad():
        for data, target in test_loader:
            print("test_2")
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
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
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
    parser.add_argument('--T', type=int, default=100, metavar='N',
                        help='SNN time window')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    #device = torch.device("cpu")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader , val_dataset= data_loader()

    #snn_dataset = SpikeDataset(val_dataset, T = args.T)
    snn_dataset = SpikeDataset(val_dataset, T = args.T, theta = max_1-0.001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=10, shuffle=False)

    #model1 = models.vgg19_bn(pretrained=True)
    #torch.save(model1.state_dict(), "imagevgg19bn_o.pt")
    #for param_tensor in model1.state_dict():
    #    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())



    model = VGG('repvgg_',bias = True).cuda()
    #image_vggrepb3
    model.load_state_dict(torch.load("ImageNet_75444.pt"), strict=False)
    snn_model = CatVGG('repvgg_', args.T, bias = True).to(device)
    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())

    #test_(model, device, val_loader)
    #torch.save(model1.state_dict(), "imageNmybn19_c_d_2_new.pt")
    correct_ = 0
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum = 0.9,weight_decay= 1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #test_(model, device, val_loader)
    k = 0
    correct = 0
    for epoch in range(1, args.epochs + 1):
        #if k<10:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-3,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        #elif k<20:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-4,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        #else:
        #    optimizer = optim.SGD(model.parameters(), lr=1e-5,momentum = 0.9,weight_decay= 0.0001)
        #    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        train(args, model, device, train_loader, optimizer, epoch)
        correct = test(model, device, val_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model.state_dict(), "image_Repvgg_1_1_cq.pt")
        k+=1
        scheduler.step()

    #fuse_module(model)
    model = fuse_bn_recursively(model)
    transfer_model(model, snn_model)
    print("SNN")
    test_(snn_model, device, snn_loader)

    #torch.save(model.state_dict(), "imageNmybn11_c_d.pt")
    #torch.save(model1.state_dict(), "imageNmybn19_c_d_2_new_4_1218_.pt")

    #correct = test_(model1, device, val_loader)

    #model = fuse_bn_recursively(model)
    #torch.save(model.state_dict(), "imageNmybn19_c_d_2_fused.pt")
    #transfer_model(model, snn_model)
    #print("successful transfer")
    #test_(snn_model, device, snn_loader)

    #test(snn_model, device, snn_loader)
    #if args.save_model
    #torch.save(model.state_dict(), "YOUR MOERL HERE.pt")
    
    
if __name__ == '__main__':
    main()

