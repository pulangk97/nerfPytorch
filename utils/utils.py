import torch

import numpy as np

import math


# comput rayo rayd to flatpoints between near-far .random sample numSam points
# rayO [B,H,W,3] rayD [B,H,W,3] 
# flatPoints [B*H*W,3]
def getflatpoints(rayO,rayD,near,far,numSam):
    rundevice=rayO.device
    #随机采样稀疏点
    tLayer=torch.linspace(near,far,numSam,device=rundevice)
    rayT=tLayer+torch.rand(numSam,device=rundevice)*((far-near)/numSam)
    points = rayO[...,None,:]+rayD[...,None,:]*rayT[:,None]

    flatPoints=torch.reshape(points,(-1,points.shape[-1]))

    return flatPoints


## split a img to random img
## img [H,W,3] rayo [H,W,3] rayd [H,W,3] 
## bi-oi-di [sqrt(N_rand),sqrt(N_rand),3]  twodimindis [xindis,yindis] xindis-yindis [N_rand]
def randbatch(img,rayo,rayd,N_rand=200*200):
    H,W,C =img.shape

    indis=torch.randint(1,H*W,(N_rand,))
    xindis=(indis%H)-1
    xindis[xindis==-1]=H-1
    yindis=np.ceil(indis/H).to(int)-1
    
    bi=img[xindis,yindis].reshape(np.int32(np.sqrt(N_rand)),np.int32(np.sqrt(N_rand)),-1)
    oi=rayo[xindis,yindis].reshape(np.int32(np.sqrt(N_rand)),np.int32(np.sqrt(N_rand)),-1)
    di=rayd[xindis,yindis].reshape(np.int32(np.sqrt(N_rand)),np.int32(np.sqrt(N_rand)),-1)
    twodimindis=[xindis.reshape(np.int32(np.sqrt(N_rand)),np.int32(np.sqrt(N_rand))),yindis.reshape(np.int32(np.sqrt(N_rand)),np.int32(np.sqrt(N_rand)))]
    return bi,oi,di,twodimindis

# def getRandompatch(img,rayo,rayd,scale=0.5):
#     H,W =img.shape[0:2]
#     randcorx=(torch.rand(1)*(H-1)).to(int) 
#     randcory=(torch.rand(1)*(W-1)).to(int)
#     dh=int(H*scale)
#     dw=int(W*scale)
#     maxcorx=H-1 if randcorx+dh-1>H-1 else randcorx+dh-1
#     maxcory=W-1 if randcory+dw-1>W-1 else randcory+dw-1

#     return img[randcorx:maxcorx,randcory:maxcory,...],rayo[randcorx:maxcorx,randcory:maxcory,...],rayd[randcorx:maxcorx,randcory:maxcory,...]



# get a img stack ray 
# pose [B,4,4]
# ray_o ray_d [B,H,W,3]
def getRayall(pose,H,W,pixSize,Focus):
    rundevice=pose.device
    xs = torch.linspace(-W/2*pixSize, W/2*pixSize, steps=W,dtype=float,device=rundevice)
    ys = torch.linspace(-H/2*pixSize, H/2*pixSize, steps=H,dtype=float,device=rundevice)
    camLoc=torch.meshgrid(xs,ys)
    dir=torch.zeros([W,H,3],device=rundevice)
    dir[:,:,0]=camLoc[1]/Focus
    dir[:,:,1]=-camLoc[0]/Focus
    dir[:,:,2]=-1
    # print(pose[np.newaxis,np.newaxis,:3,:3].shape)
    ray_d=torch.matmul(pose[:,np.newaxis,np.newaxis,:3,:3],dir[np.newaxis,...,np.newaxis])
    ray_o=pose[:,np.newaxis,np.newaxis,:3,-1].broadcast_to(pose.shape[0],W,H,3)
    return ray_o,ray_d[...,0]    

# get a single img ray 
# pose [4,4] 
# ray_o ray_d [B,H,W,3]
def getRay(pose,H,W,pixSize,Focus):
    rundevice=pose.device
    xs = torch.linspace(-W/2*pixSize, W/2*pixSize, steps=W,dtype=float,device=rundevice)
    ys = torch.linspace(-H/2*pixSize, H/2*pixSize, steps=H,dtype=float,device=rundevice)
    camLoc=torch.meshgrid(xs,ys)
    dir=torch.zeros([W,H,3],device=rundevice)
    dir[:,:,0]=camLoc[1]/Focus
    dir[:,:,1]=-camLoc[0]/Focus
    dir[:,:,2]=-1
    # print(pose[np.newaxis,np.newaxis,:3,:3].shape)
    ray_d=torch.matmul(pose[np.newaxis,np.newaxis,:3,:3],dir[...,np.newaxis])
    ray_o=pose[:3,-1].broadcast_to(W,H,3)
    return ray_o,ray_d[...,0]

# comput voxel range
# flatpoints [B,3] 
# voxel [2,3] voxel min [0,..] voxel max [1,...]
def computVoxelRange(flatpoints):
    voxel=torch.zeros(2,3)
    voxel[0,0]=torch.min(flatpoints[...,0])
    voxel[1,0]=torch.max(flatpoints[...,0])
    voxel[0,1]=torch.min(flatpoints[...,1])
    voxel[1,1]=torch.max(flatpoints[...,1])
    voxel[0,2]=torch.min(flatpoints[...,2])
    voxel[1,2]=torch.max(flatpoints[...,2])

    return voxel

