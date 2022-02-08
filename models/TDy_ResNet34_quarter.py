#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import PreEmphasis

TEMP = 31
KK = 6

print('#############################')
print(' TDy_ResNet34_quarter loaded' )
print('       Temp = %d'%(TEMP)      )
print('          K = %d'%(KK)        )
print('#############################')


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNet(BasicBlock_DY, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model

class ResNet(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNet, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input
        self.outmap_size = n_mels

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.outmap_size = int(self.outmap_size/2)
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=1)

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)
                )

        outmap_size = int(n_mels/8)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion * outmap_size, num_filters[3] * block.expansion * outmap_size)
            self.attention = self.new_parameter(num_filters[3] * block.expansion * outmap_size, 1)
            out_dim = num_filters[3] * block.expansion * outmap_size
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion * outmap_size, num_filters[3] * block.expansion * outmap_size)
            self.attention = self.new_parameter(num_filters[3] * block.expansion * outmap_size, 1)
            out_dim = num_filters[3] * block.expansion * outmap_size * 2
        elif self.encoder_type == "AVG":
            self.sap_linear = nn.AdaptiveAvgPool1d(1)
            out_dim = num_filters[3] * block.expansion * outmap_size
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                Dynamic_conv2d(self.inplanes, planes * block.expansion, self.outmap_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.outmap_size, stride, downsample))
        self.inplanes = planes * block.expansion
        if stride != 1:
            self.outmap_size = int(self.outmap_size/2)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.outmap_size))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.kaiming_normal_(out)
        return out

    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.reshape(x.size()[0],-1,x.size()[-1])

        if self.encoder_type == "SAP":
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = x.permute(0,2,1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)
        elif self.encoder_type == "AVG":
            x = self.sap_linear(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


################ Module ####################

class BasicBlock_DY(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, map_size, stride=1, downsample=None):
        super(BasicBlock_DY, self).__init__()

        self.conv1 = Dynamic_conv2d(inplanes, planes, map_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            map_size = int(map_size/2)

        self.conv2 = Dynamic_conv2d(planes, planes, map_size, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class attention2d(nn.Module):
    def __init__(self, in_planes, map_size, kernel_size, stride, padding, K, temperature):
        super(attention2d, self).__init__()
        assert temperature%3==1

        start_planes = in_planes * map_size
        hidden_planes = int(start_planes / 8)

        self.fc1 = nn.Conv1d(start_planes, hidden_planes, kernel_size, stride = stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(hidden_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden_planes, K, 3, padding=1, bias=True)
        self.temperature = temperature
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = x.reshape(x.size()[0],-1,x.size()[-1])
        
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, map_size, kernel_size, stride=1, padding=0, bias=False, K=KK, temperature=TEMP):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else : 
            self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else : 
            self.stride = stride
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else : 
            self.padding = padding

        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, map_size, self.kernel_size[-1], self.stride[-1], self.padding[-1], K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

        for i in range(self.K):
            nn.init.kaiming_normal_(self.weight[i])    

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size[0], self.kernel_size[1])

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)


        output = output.view(batch_size, self.K, self.out_planes, output.size(-2), output.size(-1))

        assert softmax_attention.shape[-1] == output.shape[-1]
        
        output = torch.einsum('bkcft,bkt->bcft', output, softmax_attention)

        return output


