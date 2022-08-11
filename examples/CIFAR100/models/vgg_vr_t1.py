
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import catSNN,catCuda
T_reduce = 8
timestep = 10
timestep_f = 10
min_1 = 0
max_1 = T_reduce/timestep
#max_1 = 1

cfg = {
    'o' : [128,128,'M',256,256,'M',512,512,'M',(1024,0),'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'o_low' : [(128,1,6.5162),(128,1,1.0253),'M',(256,1,1.6196),(256,1,0.8609),'M',(512,1,1.8470),(512,1,0.7465),'M',(1024,0,1.7244),'M'],
    'o_low_' : [(128,1,3.3402),(128,1,1.3331),'M',(256,1,1.1860),(256,1,0.8403),'M',(512,1,1.8164),(512,1,0.7146),'M',(1024,0,1.1968),'M'],


}
def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    # t=1，2: T_reduce/timestep-0.001; 
    out = catCuda.getSpikes(out, T_reduce/timestep-0.001)
    return out

def create_spike_input_cuda_low(input,T,thre):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, (max_1-0.001)/thre)
    return out

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.div(torch.floor(torch.mul(input, timestep_f)), timestep_f)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output)

quan_my = STEFunction.apply

class Clamp_q_(nn.Module):
    def __init__(self, min=min_1, max=max_1):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clamp(x, min=min_1, max=max_1)
        x = quan_my(x)
        return x

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=0.95):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=1.0)
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class NewSpike(nn.Module):
    def __init__(self, T = 4):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x
class NewSpike_low(nn.Module):
    def __init__(self, T = T_reduce, threshold = 1):
        super(NewSpike_low, self).__init__()
        self.T = T
        self.threshold = threshold

    def forward(self, x):

        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda_low(x, self.T,self.threshold)
        return x
class VGG_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=False):
        super(VGG_, self).__init__()
        #self.quantize_factor = quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(1024, 100, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),nn.Dropout2d(0.15)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),nn.BatchNorm2d(out_channels), catSNN.Clamp(max = self.clamp_max),nn.Dropout2d(0.15)]
                in_channels = out_channels


        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_5(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=False):
        super(VGG_5, self).__init__()
        #self.quantize_factor = quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(1024, 100, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_()]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),nn.BatchNorm2d(out_channels), Clamp_q_(),nn.Dropout2d(0.15)]
                in_channels = out_channels


        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_5_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=False):
        super(VGG_5_, self).__init__()
        #self.quantize_factor = quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(1024, 100, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 18

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_()]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),nn.BatchNorm2d(out_channels), Clamp_q_(),nn.Dropout2d(0.15)]
                in_channels = out_channels


        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_19(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=False):
        super(VGG_19, self).__init__()
        #self.quantize_factor = quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(512, 100, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),nn.Dropout2d(0.15)]
            else:
                padding = x[1] if isinstance(x, tuple) else 1
                out_channels = x[0] if isinstance(x, tuple) else x
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),nn.BatchNorm2d(out_channels), Clamp(),nn.Dropout2d(0.15)]
                in_channels = out_channels


        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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
        out = self.features(x)
        out = self.classifier1(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 18
        for x in cfg:
            if x == 'M':
                #layers += [self.snn.pool(2),NewSpike(self.T )]
                layers += [self.snn.pool(2),self.snn.spikeLayer(T_reduce/timestep-0.001)]

            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(T_reduce/timestep-0.001),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    #NewSpike(self.T )
                    #nn.spikeLayer(T_reduce/timestep-0.0001)
                    padding = x[1] if isinstance(x, tuple) else 1
                    out_channels = x[0] if isinstance(x, tuple) else x
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               self.snn.spikeLayer(T_reduce/timestep-0.001),nn.Dropout2d(0)]
                    #layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                    #           NewSpike(self.T ),nn.Dropout2d(0)]
                    in_channels = out_channels
        return nn.Sequential(*layers)

class CatVGG_o_low(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG_o_low, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier1 = self.snn.dense((1, 1, 1024), 100,bias = True)

    def forward(self, x):
        
        out = self.features(x)
        out = self.classifier1(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 18
        for x in cfg:
            if x == 'M':
                #IF
                #layers += [self.snn.pool(2),self.snn.spikeLayer(1.0)]
                #ASG
                layers += [self.snn.pool(2),NewSpike(self.T)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(T_reduce/timestep-0.001),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    #padding = 1
                    padding = x[1] 
                    out_channels = x[0] 
                    threshold_scaling = x[2]
                    #IF
                    #layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                    #           self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    # ASG
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               NewSpike_low(self.T,threshold_scaling),nn.Dropout2d(0)]
                    in_channels = out_channels
        return nn.Sequential(*layers)
