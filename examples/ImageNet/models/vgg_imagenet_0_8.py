
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import catSNN
import catCuda
#vgg19_pretrain_11_0429t1
T_reduce = 200
timestep = 250
timestep_f = 250.0
min_1 = 0
max_1 = T_reduce/timestep
class AddQuantization(object):
    def __init__(self, min=0., max=1.):
        self.min = min
        self.max = max
        
    def __call__(self, tensor):
        #return torch.div(torch.floor(torch.mul(tensor, timestep_f)), timestep_f)
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, timestep)), timestep),min=min_1, max=max_1)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M']
}
def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, max_1-0.0001)
    return out

class NewSpike(nn.Module):
    def __init__(self, T = 16):
        super(NewSpike, self).__init__()
        self.T = T

    def forward(self, x):
        x = (torch.sum(x, dim=4))/self.T
        x = create_spike_input_cuda(x, self.T)
        return x
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
    def __init__(self, min=0.0, max=1.0):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = quan_my(x)
        #print(x.grad)
        #x = torch.div(torch.floor(torch.mul(x, timestep_f)), timestep_f)
        return x

class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

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




class VGG1(nn.Module):
    def __init__(self, vgg_name, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG1, self).__init__()
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.relu = Clamp_q_()
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier0 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        self.classifier3 = nn.Linear(4096, 4096, bias=True)
        self.classifier6 = nn.Linear(4096, 1000, bias=True)
        self.features.apply(initialize_weights)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier0(out)
        out = self.relu(out)
        out = self.classifier3(out)
        out = self.relu(out)
        out = self.classifier6(out)
        return out
 


    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        for x in cfg:
            # [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_()]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),nn.BatchNorm2d(x),
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
        self.classifier0 = self.snn.dense((7,7,512),4096, bias=True)
        self.classifier3 = self.snn.dense((1,1,4096),4096, bias=True)
        self.classifier6 = self.snn.dense((1,1,4096),1000, bias=True)

    def forward(self, x):
        out = self.features(x)
        #out = self.snn.sum_spikes(out)/self.T
        #out = out.view(out.size(0), -1)
        out = self.classifier0(out)
        out = (torch.sum(out, dim=4))/self.T
        out = create_spike_input_cuda(out, self.T)

        out = self.classifier3(out)
        out = (torch.sum(out, dim=4))/self.T
        out = create_spike_input_cuda(out, self.T)

        out = self.classifier6(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2), NewSpike(self.T)]
            else:
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                        NewSpike(self.T)]
                in_channels = x
        # layers += [self.snn.sum_spikes_layer()]
        return nn.Sequential(*layers)

class CatVGG_t(nn.Module):
    def __init__(self, vgg_name, T, bias=True):
        super(CatVGG_t, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias

        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10, bias=False)
        #((1,1,512),10)
        self.classifier1 = self.snn.dense((7,7,512),1000, bias=True)
        #self.classifier2 = self.snn.dense(4096,4096, bias=self.bias)
        #self.classifier3 = self.snn.dense(4096,1000, bias=self.bias)

    def forward(self, x):
        out = self.features(x)
        #out = self.snn.sum_spikes(out)/self.T
        #out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        #out = self.classifier2(out)
        #out = self.classifier3(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(2)]
            else:
                layers += [self.snn.conv(in_channels, x, kernelSize=3, padding=1, bias=self.bias),
                        self.snn.spikeLayer()]
                in_channels = x
        # layers += [self.snn.sum_spikes_layer()]
        return nn.Sequential(*layers)
        
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 =  nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(outchannel)
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True)
                #nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = torch.clamp(self.conv1(x), min=0, max=1)
        out = self.conv2(out)
        out += self.shortcut(x)
        out =  torch.clamp(out, min=0, max=1)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1000):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(8*8*512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.conv1(x))
        out = torch.clamp(self.conv1(x), min=0, max=1)

        #out = nn.ReLU(self.bn1((self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResidualBlockCat(nn.Module):
    def __init__(self, inchannel, outchannel, T,stride=1):
        super(ResidualBlockCat, self).__init__()
        self.T = T
        self.snn = catSNN.spikeLayer(T)
        self.conv1 = self.snn.conv(inchannel, outchannel, kernelSize=3, stride=stride,padding=1, bias=True)
        self.conv2 =  self.snn.conv(outchannel, outchannel, kernelSize=3, stride=1,padding=1, bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                self.snn.conv(inchannel, outchannel, kernelSize=1, stride=stride, bias=True)
            )

    def forward(self, x):

        out = self.snn.spike(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.snn.spike(out)
        return out
        
class CatRes(nn.Module):
    def __init__(self, ResidualBlockCat, T,  bias=True):
        super(CatRes, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.bias=bias
        self.inchannel = 64
        self.conv1 = self.snn.conv(3, 64, kernelSize=3, stride=1,padding=1, bias=self.bias)
        self.layer1 = self.make_layer(ResidualBlockCat, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlockCat, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlockCat, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlockCat, 512, 2, stride=2)
        self.pool1 = self.snn.pool(4)
        self.fc = self.snn.dense((8,8,512),1000,bias=self.bias)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, self.T, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
  

    def forward(self, x):
        out = self.snn.spike(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.snn.spike(self.pool1(out))
        out = self.snn.spike(self.fc(out))
        #print(self.snn.sum_spikes(out))
        out = self.snn.sum_spikes(out)/self.T
        return out
        


def test():
    net = CatVGG('VGG11', 60)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
