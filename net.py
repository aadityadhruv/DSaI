import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np




def inception(in_size,out_size,k1=100,k2=100):

    l_s = out_size // 4
    return nn.ModuleList([
        nn.Sequential(nn.Conv2d(in_size,k1,1), nn.ReLU(), nn.Conv2d(k1,l_s,3,dilation=1,stride=3), nn.ReLU()),
        nn.Sequential(nn.Conv2d(in_size,k2,1), nn.ReLU(), nn.Conv2d(k2,l_s,5,stride=3,padding=1,dilation=1), nn.ReLU()),
        nn.Sequential(nn.MaxPool2d(3), nn.Conv2d(in_size,l_s,1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(in_size,l_s,5,stride=3,padding=1,dilation=1), nn.ReLU())])




class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        

        self.norm3 = nn.BatchNorm2d(3)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm256 = nn.BatchNorm2d(256)
        self.norm512 = nn.BatchNorm2d(512)
        self.norm892 = nn.BatchNorm2d(892)
        self.norm1000 = nn.BatchNorm2d(1000)
        self.pool = nn.MaxPool2d(2)

        #227x227
        self.stem = nn.Sequential(
            nn.Conv2d(3,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            
            nn.Conv2d(32,192,1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        #54x54


        self.inc1 = inception(256,512, k1=192,k2=192)
        #18x18
        self.inc2 = inception(512,892, k1=367,k2=367)
        #6x6

        self.restem = nn.Sequential(
            nn.Conv2d(892,1024,5,padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(1024,1024,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        

        #self.inc3 = inception(1024,,k1=571,k2=571)
        #1x1
        #self.inc4 = inception(892,1000,k1=571,k2=571)




        self.fc1 = nn.Linear(512,192)
        self.fc2 = nn.Linear(192, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)


        
        
        



    def forward(self, x):
        #print(type(x))
        original_dim = x.shape


        x = self.norm3(x)
        x = self.stem(x)
        #54x54

        x = torch.cat((self.inc1[0](x),self.inc1[1](x), self.inc1[2](x), self.inc1[3](x)), 1)
        x = self.norm512(x)
        x = torch.cat((self.inc2[0](x),self.inc2[1](x), self.inc2[2](x), self.inc2[3](x)), 1)
        x = self.norm892(x)


        x = self.restem(x)
        #1x1

        x = torch.reshape(x,(original_dim[0],-1))
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        

        return x