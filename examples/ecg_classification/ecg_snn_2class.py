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
from sklearn.metrics import confusion_matrix
import numpy as np
import time

T = 50
timestep = 50
T_win = 50
T_reduce = 50
min_1 = 0
max_1 = T_win/T
import torch
from torch.utils.data import Dataset
#torch.manual_seed(3407)

class CustomDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.data = data['data']
        self.labels = data['label']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



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
        x = torch.clamp(torch.div(torch.floor(torch.mul(torch.tensor(tensor), T)), T),min=min_1, max=max_1)
        x = x.reshape(1,x.shape[0],1)
        x = x.float()
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

        self.conv1 = nn.Conv2d(1, 32, (3,1), 1,(0,0), bias=True)
        self.Bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3,1), 1,(0,0), bias=True)
        self.Bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(88*32, 2, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        x = self.conv2(x)
        x = self.Bn2(x)
        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)
        
        x = F.avg_pool2d(x, (2,1))
        x = torch.clamp(x, min=min_1, max=max_1)
        x = Quantization_(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn

        self.conv1 = snn.conv(1, 32, (3,1), 1,(0,0), bias=True)
        self.conv2 = snn.conv(32, 32, (3,1), 1,(0,0), bias=True)
        self.pool = snn.pool((2,1))
     
        self.fc1 = snn.dense((1,88,32), 2, bias=True)


    def forward(self, x):
    
        
        y = self.conv1(x)
        #y= self.snn.spike(y,0.99)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        y = self.conv2(y)
        #y= self.snn.spike(y,0.99)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)
        
        y = self.pool(y)
        #y= self.snn.spike(y,0.99)
        y = (torch.sum(y, dim=4)) /self.T
        y = create_spike_input_cuda(y, self.T)

        x = self.fc1(y)


        
        return self.snn.sum_spikes(x)/self.T


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight_neg, weight_pos, weight_fn):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight_neg = weight_neg
        self.weight_pos = weight_pos
        self.weight_fn = weight_fn  

    def forward(self, inputs, target):
        loss = F.cross_entropy(inputs, target, reduction='none')
        predicted = torch.argmax(inputs, dim=1)
        weights = torch.where(target == 1, 
                              torch.where(predicted == target, self.weight_pos, self.weight_fn),
                              self.weight_neg)

        weighted_loss = loss * weights

        return torch.mean(weighted_loss)
        
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    weight_neg = 1.0  
    weight_pos = 2  
    weight_fn = 10
    criterion = WeightedCrossEntropyLoss(weight_neg, weight_pos,weight_fn)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 2)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.cross_entropy(output, target)
        #loss = F.mse_loss(output, onehot.type(torch.float))
        loss = criterion(output, target)

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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 2)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return cm,correct

def test_createdata(model, device, test_loader, four_class_dataset):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    class_one_indices = []  # 存储识别为类别 1 的样本索引

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 2)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # 记录识别为类别 1 的样本索引
            class_one_indices.extend(batch_idx * test_loader.batch_size + i for i in pred.cpu().numpy().nonzero()[0])

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # 提取四分类中对应的数据和标签
    class_one_data = [four_class_dataset.dataset[i][0].numpy() for i in class_one_indices]
    class_one_labels = [four_class_dataset.dataset[i][1] for i in class_one_indices]

    # 保存为新的 npz 文件
    np.savez('4class_abnormal.npz', data=class_one_data, labels=class_one_labels)

    # 计算混淆矩阵并输出结果
    

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

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

    torch.manual_seed(3407)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform_train = transforms.Compose([
        AddQuantization()
        ])

    transform=transforms.Compose([
        AddQuantization()
        ])
    

    train_dataset = CustomDataset('train_nor_binary.npz', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)

    test_dataset = CustomDataset('test_nor_binary.npz', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    test_dataset_4 = CustomDataset('test_nor.npz', transform=transform)
    test_loader_4 = torch.utils.data.DataLoader(test_dataset_4, batch_size=100, shuffle=False)

    val_dataset = CustomDataset('val_nor_binary.npz', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)

    
    snn_dataset = SpikeDataset(test_dataset, T = args.T,theta = max_1-0.01)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=100, shuffle=False)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)

    model.load_state_dict(torch.load("weight/2_class.pt"), strict=False)

    test(model, device, train_loader)
    test(model, device, val_loader)
    test(model, device, test_loader)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    max_li =10000 
    for ll in range(1000):
        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        ACC = 0

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, train_loader)
            ACC_m ,Acc_ = test(model, device, val_loader)
            test(model, device, test_loader)
            
            if Acc_/7537>0.5:
                if ACC_m[1][0]< max_li:
                    max_li = ACC_m[1][0]
                    print("----------------------------------------------------")
                    print(max_li,Acc_)
                    name = "weight/wT50"+str(ll)+str(epoch)+".pt"
                    torch.save(model.state_dict(), str(name))
            scheduler.step()

    fuse_module(model)
    transfer_model(model, snn_model)
    test_createdata(snn_model, device, snn_loader,test_loader_4)
    #if args.save_model:



if __name__ == '__main__':
    main()
