
from asyncio import futures
import torch

import numpy as np

import math
import utils.encoding as encoding
import utils.utils as ut

def randomRayt(near,far,numSam=64):
    tLayer=torch.linspace(near,far,numSam,device=torch.device('cpu'))
    rayT=tLayer+torch.rand(numSam,device=torch.device('cpu'))*((far-near)/numSam)    
    return rayT
## compute coast points in cpu
## rayO [H,W,3] rayD [H,W,3] 
## return flatPoints [H*W*numSam,3] flatrayd [H*W*numSam,3] rayT [numSam] H W
def getcoastpoints(rayT,rayO,rayD):

    rundevice=rayO.device
    if rayT.device!=rundevice:
        rayT=rayT.to(rundevice)

    points = rayO[...,None,:]+rayD[...,None,:]*rayT[:,None]
    flatPoints=torch.reshape(points,(-1,points.shape[-1]))
    flatrayd=torch.reshape(rayD[...,None,:].broadcast_to(points.shape),(-1,rayD.shape[-1]))

    return flatPoints,flatrayd


# compute fine points in cpu
## rayO [H,W,3] rayD [H,W,3] sigma [H,W,numcoast] rayT [numcoast]
## return flatPoints [H*W*(numSam+numFine),3] flatrayd [H*W*(numSam+numFine),3] newRayT [H,W,(numSam+numFine)] H W
def getfinepoints(rayT,sigma,rayO,rayD,numFine=64):
    rundevice=rayO.device
    if rayT.device!=rundevice:
        rayT=rayT.to(rundevice)

    dt=torch.concat([torch.diff(rayT),torch.tensor(1e10,device=rundevice).broadcast_to(1)])
    if sigma.device!=rundevice:
        sigma=sigma.to(rundevice)
    else:
        sigma=sigma
    alfa=torch.exp(-sigma*dt)

    T=torch.cumprod(alfa + 1e-10,dim=-1,dtype=torch.float32)
    weight=T*(1.-alfa)  #权重计算，防止除0
    weight=weight[...,1:-1]+1e-5
    weight=weight/(torch.sum(weight,dim=-1,dtype=torch.float32)[...,np.newaxis])  #权重归一化 


          
    cumWeight=torch.concat((torch.zeros_like(weight,dtype=torch.float32,device=rundevice)[...,:1],torch.cumsum(weight,dim=-1,dtype=torch.float32)),dim=-1)  #权重累积，权重大对应选取采样点个数多


    

    indis=torch.sort(torch.rand((weight.shape[0],weight.shape[1],(rayT.shape[0]+numFine)),dtype=torch.float32,device=rundevice),dim=-1).values #获取随机权重

    rayT_mid=(rayT[1:]+rayT[:-1])/2
    
    sortedIndis=torch.searchsorted(cumWeight,indis,right=True)#为每个权重值分配点，权重大对应选取采样点个数多，且权重间距正比于实际距离

    downIndis=torch.max(torch.tensor(0,device=rundevice),sortedIndis-1)

    upIndis=torch.min(sortedIndis,torch.tensor(cumWeight.shape[-1]-1,device=rundevice))

    scaleIndis=torch.stack([downIndis,upIndis],-1)


    w_g=torch.gather(cumWeight[...,np.newaxis,:].broadcast_to(cumWeight.shape[0],cumWeight.shape[1],scaleIndis.shape[-2],cumWeight.shape[-1]),-1,scaleIndis)
 
    rt_g=torch.gather(rayT_mid[np.newaxis,np.newaxis,np.newaxis,:].broadcast_to(scaleIndis.shape[0],scaleIndis.shape[1],scaleIndis.shape[2],rayT_mid.shape[-1]),-1,scaleIndis)

    scale=w_g[...,1]-w_g[...,0]
    scale[scale<1e-5]=1
    t=(indis-w_g[...,0])/scale
    newRayT=rt_g[...,0]+t*(rt_g[...,1]-rt_g[...,0]) #对于每个新的采样点，精采样点的位置正比于该点权重所在位置
    # print(newRayT.shape)
    newRayT=torch.sort(torch.concat((newRayT,rayT[np.newaxis,np.newaxis,:].broadcast_to((newRayT.shape[0],newRayT.shape[1],rayT.shape[-1]))),dim=-1),dim=-1).values

    points = rayO[...,None,:]+rayD[...,None,:]*newRayT[...,:,None]
    flatPoints=torch.reshape(points,(-1,points.shape[-1]))
    flatrayd=torch.reshape(rayD[...,None,:].broadcast_to(points.shape),(-1,rayD.shape[-1]))

    return flatPoints,flatrayd,newRayT



def batch_enc(enc_hash,enc_freq,flatpoints,flatrayd,fn,if_hash=False,chunkSize=1024*32):
    # device=next(fn.parameters()).device
    if if_hash:
        return torch.concat([fn(enc_hash(flatpoints[i:i+chunkSize,...]),enc_freq(flatrayd[i:i+chunkSize,...])) for i in range(0,flatpoints.shape[0],chunkSize)],0)
    else:
        return torch.concat([fn(enc_freq(flatpoints[i:i+chunkSize,...]),enc_freq(flatrayd[i:i+chunkSize,...])) for i in range(0,flatpoints.shape[0],chunkSize)],0)



# comput raw to rgb in GPU
def raw2rgb(raw,H,W,numsam):
    
        modelOut=torch.reshape(raw,(H,W,numsam,-1))
        sigma=torch.relu(modelOut[...,3])
        rgb=torch.sigmoid(modelOut[...,:3])
 
        return sigma,rgb


# comput RGB to colormap depthmap accmap
# rayT [H,W,numsam] sigma [H,W,numsam] rgb [H,W,numsam,3]
def rgb2output(rayT,sigma,rgb):
    rundevice=sigma.device
    rayT=rayT.to(rundevice)
    dt=torch.concat([torch.diff(rayT,dim=-1),(torch.tensor(1e10,device=rundevice).broadcast_to(rayT.shape))[...,:1]],dim=-1)
    alfa=torch.exp(-sigma*dt)

    T=torch.cumprod(alfa + 1e-10,dim=-1,dtype=torch.float32)

    weight=T*(1.-alfa)
    rgb=rgb.permute(3,0,1,2)
    cr=torch.sum(weight*rgb,dim=-1)
    depthMap=torch.sum(weight*rayT,dim=-1)
    accMap=torch.sum(weight,dim=-1)
    return cr.permute(1,2,0),depthMap,accMap


