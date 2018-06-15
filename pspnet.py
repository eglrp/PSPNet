''' 
psp net 
'''
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

class SppBlock(nn.Module):
    def __init__(self,level,in_channel=256):
        super().__init__()
        self.level=level
        self.convblock=nn.Sequential(nn.Conv2d(in_channel,512,1,1),
                                     nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        
    def forward(self,x):
        size=x.shape[2:]
        x=F.adaptive_avg_pool2d(x,output_size=(self.level,self.level))# average pool
        x=self.convblock(x)
        x=F.upsample(x,size=size,mode='bilinear',align_corners=True)
        return x
        

class SPP(nn.Module):
    def __init__(self,in_channel=256):
        super().__init__()
        self.spp1=SppBlock(level=1,in_channel=in_channel)
        self.spp2=SppBlock(level=2,in_channel=in_channel)
        self.spp3=SppBlock(level=3,in_channel=in_channel)
        self.spp6=SppBlock(level=6,in_channel=in_channel)
        
    def forward(self,x):
        x1=self.spp1(x)
        x2=self.spp2(x)
        x3=self.spp3(x)
        x6=self.spp6(x)
        out=torch.cat([x,x1,x2,x3,x6],dim=1)
        return out
'''
可以任意更改 in_channel，因为你想选择不同层网络layer1 or layer2 ... 。也可以fine-tune
'''
class PSPNet(nn.Module):
    def __init__(self,class_number=5):
        super().__init__()
        encoder=torchvision.models.resnet50(pretrained=True)################
        self.start=nn.Sequential(encoder.conv1,encoder.bn1,encoder.relu)
        self.maxpool=encoder.maxpool
        self.layer1=encoder.layer1#256
        self.layer2=encoder.layer2#512
        self.layer3=encoder.layer3#1024
        self.layer4=encoder.layer4#2048
        self.spp=SPP(in_channel=2048)#############
        
        self.score_s=nn.Sequential(nn.Conv2d(512*5,512,3,1,padding=1),nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))
        self.score=nn.Conv2d(512,class_number,1,1)# no relu
        
        
    def forward(self,x):
        size=x.shape[2:]
        x=self.start(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.spp(x)
        
        x=self.score_s(x)
        x=F.upsample(x,size=size,mode='bilinear',align_corners=True)
        x=self.score(x)
        
        return x
