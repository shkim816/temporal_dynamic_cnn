#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import time

TEMP = 31
KK = 6

print('#############################')
print('      TDy_VGGVox loaded      ')
print('        Temp = %d'%(TEMP)     )
print('           K = %d'%(KK)       )
print('#############################')

class MainModel(nn.Module):
    def __init__(self, nOut = 1024, encoder_type='TAP', log_input=True, **kwargs):
        super(MainModel, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.encoder_type = encoder_type
        self.log_input    = log_input

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5,7), stride=(1,2), padding=(2,2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))
        )

        self.layer2 = nn.Sequential(
            Dynamic_conv2d(96, 256, 64, kernel_size=(5,5), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        )

        self.layer3 = nn.Sequential(
            Dynamic_conv2d(256, 384, 15, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            Dynamic_conv2d(384, 256,15, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            Dynamic_conv2d(256, 256,15, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        )

        self.layer6 = nn.Sequential(
            Dynamic_conv2d(256, 512, 7, kernel_size=(7,1), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        if self.encoder_type == "MAX":
            self.encoder = nn.AdaptiveMaxPool2d((1,1))
            out_dim = 512
        elif self.encoder_type == "AVG":
            self.encoder = nn.AdaptiveAvgPool2d((1,1))
            out_dim = 512
        elif self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(512, 512)
            self.attention = self.new_parameter(512, 1)
            out_dim = 512
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        self.instancenorm   = nn.InstanceNorm1d(64)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=64)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
        
    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        if self.encoder_type == "MAX" or self.encoder_type == "AVG":
            x = self.encoder(x)
            x = x.view((x.size()[0], -1))

        elif self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)

        x = self.fc(x)

        return x

################ Module ####################

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


