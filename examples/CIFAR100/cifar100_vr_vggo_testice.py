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

sample_tensor = torch.rand(3, 3,3)  # 例如，100x100 的随机张量

# 初始化您的类并应用它
quantizer = AddQuantization_new()
x = quantizer(sample_tensor)
print(x)
# 将张量转换为一维数组
x_flattened = x.flatten()
print(x_flattened)
# 找出所有唯一的值
unique_values = set(x_flattened.tolist())

# 打印唯一值
print("Unique values in x:", unique_values)

for i in range(1,T_reduce+1):
    print(i)