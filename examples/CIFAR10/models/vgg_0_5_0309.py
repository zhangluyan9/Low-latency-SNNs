
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import catSNN
import catCuda
import torch.nn.functional as F
import math
T_reduce = 2
timestep = 3
timestep_f = 3
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
def compute_k(point1,point2,p):
    k = (point2-point1)/(math.pow(point2,p)-math.pow(point1,p))
    return k
def compute_b(point1,point2,p):
    k = (point2-point1)/(math.pow(point2,p)-math.pow(point1,p))
    b = point1-k*math.pow(point1,p)
    return b

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    # 1/2.5:max_1-0.0001; 8/10 : max_1-0.001 ; 16/20 : max_1-0.0001 ; 32/40 : max_1-0.001 ;
    out = catCuda.getSpikes(out, max_1-0.001)
    return out

def create_spike_input_cuda_low(input,T,thre):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, (max_1-0.001)/thre)
    return out

def initialize_weights(model):
    for m in model.modules():
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

class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()

    def forward(self, x):
        x = torch.clamp(x, min=min_1, max=max_1)
        b1 = (x <0.2)  & (x >= 0)
        b2 = (x >= 0.2)& (x < 0.4)
        b3 = (x >= 0.4)& (x <= 0.6)

        a1 = compute_k(0,0.2,5)*x*x*x*x*x
        a2 = compute_k(0.2,0.4,5)*x*x*x*x*x+compute_b(0.2,0.4,5)
        a3 = compute_k(0.4,0.6,5)*x*x*x*x*x+compute_b(0.4,0.6,5)

        #a3 = 25*x*x*x/19+6/19

        
        a1 = a1 * b1 
        a2 = a2 * b2
        a3 = a3 * b3
        c = a1 + a2 + a3
        #print(c.grad)
        return c
        
        #return x*x*x

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=0.95):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=1.0)

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=timestep_f):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        return F.hardtanh(grad_output), None 

Quantization_ = Quantization.apply

class NewSpike(nn.Module):
    def __init__(self, T = T_reduce):
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

class printlayer(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(printlayer, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        distri = x.reshape(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3])
        distri = distri.cpu().numpy().tolist()
        result = {}
        for i in set(distri):
            result[i] = distri.count(i)
        print("dis =", result)
        return x


class VGG_o_(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max=1.0, bias=True):
        super(VGG_o_, self).__init__()
        self.clamp_max = clamp_max
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(1024, 10, bias=True)
        self.features.apply(initialize_weights)

    def forward(self, x):
        #print(self.features[0](x)).size
        out = self.features(x)
        #print((out))
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        #turn to 18 it T = 1; ALSO, THE NAME OF LAST LAYER IS "classifier4" insted of "classifier1", PLEASE CHANGE IF YOU USE OUR PRETRAIN MODEL
        in_channels = 18

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
          

class CatVGG_o(nn.Module):
    def __init__(self, vgg_name, T, is_noise=False, bias=True):
        super(CatVGG_o, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.is_noise = is_noise
        self.bias = bias

        self.features = self._make_layers(cfg[vgg_name], is_noise)
        self.classifier1 = self.snn.dense((1, 1, 1024), 10,bias = True)

    def forward(self, x):
        
        out = self.features(x)
        out = self.classifier1(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        #turn to 18 it T = 1; ALSO, THE NAME OF LAST LAYER IS "classifier4" insted of "classifier1", PLEASE CHANGE IF YOU USE OUR PRETRAIN MODEL
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
                               self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    in_channels = x
                else:
                    #padding = 1
                    padding = x[1] 
                    out_channels = x[0] 
                    threshold_index = x[2]
                    #IF
                    #layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                    #           self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
                    # ASG
                    layers += [self.snn.conv(in_channels, out_channels, kernelSize=3, padding=padding, bias=self.bias),
                               NewSpike(self.T),nn.Dropout2d(0)]
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
        self.classifier1 = self.snn.dense((1, 1, 1024), 10,bias = True)

    def forward(self, x):
        
        out = self.features(x)
        out = self.classifier1(out)
        out = self.snn.sum_spikes(out) / self.T
        return out

    def _make_layers(self, cfg, is_noise=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                #IF
                #layers += [self.snn.pool(2),self.snn.spikeLayer(1.0)]
                #ASG
                layers += [self.snn.pool(2),NewSpike(self.T)]
            else:
                if is_noise:
                    layers += [self.snn.mcConv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                               self.snn.spikeLayer(1.0),nn.Dropout2d(0)]
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


