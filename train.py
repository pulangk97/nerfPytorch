import torch

import numpy as np

import math
import utils.multigpu as multigpu
import dataset as td
import utils.utils as utils
import utils.volumrender as render
import utils.encoding as enc

import model as NeRFmodel
import loss as lossf

## train func by nerf or hashnerf 
## input ImagTensor [B,H,W,C] posesTensor [B,4,4] focal [1] 
## NeRF : if_hash=False embeddings: None model: full MLP
## HashNeRF : if_hash=True embeddings: [embedding list] model: tiny MLP

def train(ImgTensor,posesTensor,focal,model,enc_freq=None,enc_hash=None,if_hash=False,iter_num=1000,numgpu=1,learning_rate=5e-4,numCoastsam=64,numFinesam=128,chunkSize=1024*32,N_rand=200*200):

    numpic,H, W = ImgTensor.shape[0:3] 
    lossF=torch.nn.MSELoss()

    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)   
    
      
    if if_hash :
        rayoall,raydall=utils.getRayall(posesTensor,H,W,1,focal)
        flatpoints=utils.getflatpoints(rayoall,raydall,near=2,far=6,numSam=numFinesam)
        voxel_bound_list=multigpu.datatoGPU([utils.computVoxelRange(flatpoints) for i in range(numgpu)],numgpu)
        
        ml_el,ll,opl=multigpu.modeltoGPU(model,lossF,learning_rate,numgpu,embeddings=enc_hash,if_hash=if_hash)
        ml=ml_el[0]
        emb_list=ml_el[1]

        hashcodinglist = [enc.HashEncoding(voxel_bound_list[i],embedding_input=emb_list[i]) for i in range(len(voxel_bound_list))]

    else:
        hashcodinglist=[None for i in range(numgpu)]
        ml_el,ll,opl=multigpu.modeltoGPU(model,lossF,learning_rate,numgpu,embeddings=None,if_hash=if_hash)
        ml=ml_el[0]
        emb_list=None

    # all img rayt and sigma conductor
    coastRayTAlll=torch.zeros((numpic,numCoastsam)).detach()
    # coastSigmaAlll=multigpu.splitImgToPatch(torch.zeros((numpic,H,W,numCoastsam)).permute(0,3,1,2),numgpu)[0]
    coastSigmaAlll=torch.zeros((numpic,H,W,numCoastsam)).detach()


    # for i in range(len(coastSigmaAlll)):
    #     coastSigmaAlll[i]=coastSigmaAlll[i].permute(0,2,3,1)


    for i in range(iter_num):
        
        imIndex=torch.randint(0,numpic,())
        currentImg=ImgTensor[imIndex,...].to(torch.float32)
        currentPose=posesTensor[imIndex,...]
        rayo,rayd=utils.getRay(currentPose,H,W,1,focal)

        # currentImg,rayo,rayd=utils.getRandompatch(currentImg,rayo,rayd,scale=0.5)
        currentImg,rayo,rayd,indisbatch = utils.randbatch(currentImg,rayo,rayd,N_rand=N_rand)

        rayo=rayo.permute(2,0,1)
        rayd=rayd.permute(2,0,1)
        currentImg=currentImg.permute(2,0,1)

        # split on cpu
        rayol=multigpu.datatoGPU(multigpu.splitImgToPatch(rayo,numgpu)[0],numgpu)
        raydl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[0],numgpu)
        ixl=multigpu.datatoGPU(multigpu.splitImgToPatch(indisbatch[0],numgpu)[0],numgpu)
        iyl=multigpu.datatoGPU(multigpu.splitImgToPatch(indisbatch[1],numgpu)[0],numgpu)
        splitHl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[1],numgpu)
        splitWl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[2],numgpu)

        # split gt on gpu
        labell=multigpu.datatoGPU(multigpu.splitImgToPatch(currentImg,numgpu)[0],numgpu)


        rayT=render.randomRayt(near=2,far=6,numSam=numCoastsam)
        
        rayTl=multigpu.datatoGPU([rayT for i in range(numgpu)],numgpu)

        coastRayTAlll[imIndex,...]=rayT.to(coastRayTAlll.device)


        for j in range(numgpu): 
            opl[j].zero_grad()
            # coastRayT,coastSigma,coastRGB=render.renderRayCoast(rayol[j].permute(1,2,0),raydl[j].permute(1,2,0),2,6,numSam=numCoastsam,model=ml[j],enc_hash=hashcodinglist[j],if_hash=if_hash)
            flatp,flatd=render.getcoastpoints(rayT=rayTl[j],rayD=raydl[j].permute(1,2,0),rayO=rayol[j].permute(1,2,0)) #getpoints in gpu
            
            coastraw=render.batch_enc(enc_hash=hashcodinglist[j],enc_freq=enc_freq,flatpoints=flatp,flatrayd=flatd,fn=ml[j],if_hash=if_hash,chunkSize=chunkSize)

            sigma,rgb=render.raw2rgb(coastraw,splitHl[j],splitWl[j],numCoastsam)

            colormap,depth,acc=render.rgb2output(sigma=sigma,rgb=rgb,rayT=rayTl[j])

            coastSigmaAlll[imIndex,ixl[j],iyl[j],...]=sigma.to(coastSigmaAlll.device).detach()
            # coastSigmaAlll[j][imIndex,...]=sigma.to(coastSigmaAlll[j].device)

            loss=ll[j](colormap,labell[j].permute(1,2,0))
            psnr=lossf.computPSNR(colormap,labell[j].permute(1,2,0))
            print("epoch="+str(i)+",gpu="+str(j)+",trainloss="+str(loss))
            print("epoch="+str(i)+",gpu="+str(j)+",trainpsnr="+str(psnr))
            loss.backward()


        multigpu.sendToGPU(ml,multigpu.meanParams(ml))
        if if_hash:
            multigpu.sendToGPU(emb_list,multigpu.meanParams(emb_list))
        for j in range(numgpu): 
            opl[j].step()
        


## fine train process
    for i in range(iter_num):
        
        imIndex=torch.randint(0,numpic,())
        currentImg=ImgTensor[imIndex,...].to(torch.float32)
        currentPose=posesTensor[imIndex,...]
        rayo,rayd=utils.getRay(currentPose,H,W,1,focal)

        # currentImg,rayo,rayd=utils.getRandompatch(currentImg,rayo,rayd,scale=0.5)
        currentImg,rayo,rayd,indisbatch = utils.randbatch(currentImg,rayo,rayd,N_rand=N_rand)

        rayo=rayo.permute(2,0,1)
        rayd=rayd.permute(2,0,1)
        currentImg=currentImg.permute(2,0,1)

        # split on cpu
        rayol=multigpu.datatoGPU(multigpu.splitImgToPatch(rayo,numgpu)[0],numgpu)
        raydl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[0],numgpu)
        ixl=multigpu.datatoGPU(multigpu.splitImgToPatch(indisbatch[0],numgpu)[0],numgpu)
        iyl=multigpu.datatoGPU(multigpu.splitImgToPatch(indisbatch[1],numgpu)[0],numgpu)
        splitHl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[1],numgpu)
        splitWl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[2],numgpu)

        # split gt on gpu
        labell=multigpu.datatoGPU(multigpu.splitImgToPatch(currentImg,numgpu)[0],numgpu)




        for j in range(numgpu): 
            opl[j].zero_grad()
            # coastRayT,coastSigma,coastRGB=render.renderRayFine(coastRayTAlll[j][imIndex,0,0,:],coastSigmaAlll[j][imIndex,...],rayol[j].permute(1,2,0),raydl[j].permute(1,2,0),numFine=numFinesam,model=ml[j],enc_hash=hashcodinglist[j],if_hash=if_hash)
            # flatp,flatd,rayt=render.getcoastpoints(rayD=raydl[j].permute(1,2,0),rayO=rayol[j].permute(1,2,0),near=2,far=6,numSam=numCoastsam) #getpoints in cpu
            flatPoints,flatrayd,newRayT=render.getfinepoints(coastRayTAlll[imIndex,...],sigma=coastSigmaAlll[imIndex,ixl[j],iyl[j],...],rayO=rayol[j].permute(1,2,0),rayD=raydl[j].permute(1,2,0),numFine=numFinesam)

            coastraw=render.batch_enc(enc_hash=hashcodinglist[j],enc_freq=enc_freq,flatpoints=flatPoints,flatrayd=flatrayd,fn=ml[j],if_hash=if_hash,chunkSize=chunkSize)

            sigma,rgb=render.raw2rgb(coastraw,splitHl[j],splitWl[j],newRayT.shape[-1])

            colormap,depth,acc=render.rgb2output(sigma=sigma,rgb=rgb,rayT=newRayT)

            loss=lossF(colormap,labell[j].permute(1,2,0))
            psnr=lossf.computPSNR(colormap,labell[j].permute(1,2,0))
            print("fine train epoch="+str(i)+",gpu="+str(j)+",trainloss="+str(loss))
            print("fine train epoch="+str(i)+",gpu="+str(j)+",psnrloss="+str(psnr))
            loss.backward()

        multigpu.sendToGPU(ml,multigpu.meanParams(ml))
        if if_hash:
            multigpu.sendToGPU(emb_list,multigpu.meanParams(emb_list))
        for j in range(numgpu): 
            opl[j].step()
    return ml,emb_list,coastRayTAlll,coastSigmaAlll,hashcodinglist


