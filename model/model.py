import torch

import numpy as np

# class NeRFmodel(torch.nn.Module):
#     def __init__(self,lx,ld,W):
#         super().__init__()
#         self.MLPinput=torch.nn.Linear(lx+ld+3,W)
#         self.MLPmid0=torch.nn.Sequential(*self.MLPstd(4,256))
#         self.Denseout=torch.nn.Linear(W+lx+ld+3,W)
#         self.MLPmid1=torch.nn.Sequential(*self.MLPstd(2,256))      
#         self.Modelout=torch.nn.Linear(W,4)
#         self.relu=torch.nn.ReLU()
#     def MLPstd(self,numsLayer,width):
#         mlpstd=[]
#         for i in range(numsLayer,):
#           mlpstd.append(torch.nn.Linear(width,width))
#           mlpstd.append(torch.nn.ReLU())
#         return mlpstd
#     def forward(self,X):
#         x0=self.relu(self.MLPinput(X))
#         x0=self.MLPmid0(x0)
#         x0=self.relu(self.Denseout(torch.concat([x0,X],-1)))
#         x0=self.MLPmid1(x0)
#         output=self.Modelout(x0)
#         return output
class NeRFmodel(torch.nn.Module):
    def __init__(self,lx=36,ld=36,W=256):
        super().__init__()
        self.MLPinput=torch.nn.Linear(lx+3,W)
        self.MLPmid0=torch.nn.Sequential(*self.MLPstd(4,256))
        self.Denseout=torch.nn.Linear(W+lx+3,W)
        self.MLPmid1=torch.nn.Sequential(*self.MLPstd(2,256))      
        self.relu=torch.nn.ReLU()

        self.feature_linear = torch.nn.Linear(W, W)
        self.view_linear=torch.nn.Linear(ld+ 3 + W, W//2)
        self.alpha_linear = torch.nn.Linear(W, 1)
        self.rgb_output = torch.nn.Linear(W//2, 3)


    def MLPstd(self,numsLayer,width):
        mlpstd=[]
        for i in range(numsLayer,):
          mlpstd.append(torch.nn.Linear(width,width))
          mlpstd.append(torch.nn.ReLU())
        return mlpstd


    def forward(self,embpoints,embdir):
        x0=self.relu(self.MLPinput(embpoints))
        x0=self.MLPmid0(x0)
        x0=self.relu(self.Denseout(torch.concat([x0,embpoints],-1)))
        x0=self.MLPmid1(x0)

        dense=self.alpha_linear(x0)

        x0=self.feature_linear(x0)
        x0=self.relu(self.view_linear(torch.concat((x0,embdir),dim=-1)))
        x0=self.rgb_output(x0)

        return torch.concat((x0,dense),dim=-1)
 
class NeRFtinymodel(torch.nn.Module):
    def __init__(self,W=64,ld=36,n_features_per_level=2,n_levels=16,num_hiddense=1,num_hidcolor=2,num_hidcolordim=15):
        super().__init__()
        self.inputsize=n_features_per_level*n_levels
        
        self.MLPinput=torch.nn.Linear(self.inputsize,W)
        self.MLPmid0=torch.nn.Sequential(*self.MLPstd(num_hiddense,W))
        self.Denseout=torch.nn.Linear(W,1+num_hidcolordim)
        self.colorlayerin=torch.nn.Linear(num_hidcolordim+ld+3,W)
        self.MLPmid1=torch.nn.Sequential(*self.MLPstd(num_hidcolor,W))      
        self.colorout=torch.nn.Linear(W,3)
        self.relu=torch.nn.ReLU()
    def MLPstd(self,numsLayer,width):
        mlpstd=[]
        for i in range(numsLayer,):
          mlpstd.append(torch.nn.Linear(width,width))
          mlpstd.append(torch.nn.ReLU())
        return mlpstd
    def forward(self,emb_points,enc_dir):
        x0=self.relu(self.MLPinput(emb_points))
        x0=self.MLPmid0(x0)
        x0=self.Denseout(x0)

        dense=x0[...,:1]
        h=x0[...,1:]

        h=self.relu(self.colorlayerin(torch.concat([h,enc_dir],-1)))
        h=self.MLPmid1(h)
        color=self.colorout(h)
        return torch.concat((color,dense),dim=-1)