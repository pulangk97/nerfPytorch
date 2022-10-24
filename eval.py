import copy
import torch
import utils.multigpu as multigpu
import utils.utils as ut
import utils.volumrender as render
import numpy as np
def eval_nerf_fine(model,ml,emb,emb_list,hashcodinglist,enc_freq,ImgTensor,posesTensor,focal,coastRayTAlll,coastSigmaAlll,numgpu=1,if_hash=False,savepath=None,numFinesam=128,chunkSize=1024*32):
    numpic,H,W=ImgTensor.shape[0:3]

    try:
        with torch.no_grad():
            meanparam=multigpu.meanParams(ml)
            n=0
            for para in model.parameters():
                para.copy_(meanparam[n])
                n=n+1
        torch.save(model.state_dict(),savepath+"model_fine.p")
    except:
        print("model to cpu error")



    if if_hash:
        try:
            with torch.no_grad():
                n=0
                meanparam=multigpu.meanParams(emb_list)
                for para in emb.parameters():
                    para.copy_(meanparam[n])
                    n=n+1
            torch.save(emb.state_dict(),savepath+"model_fine_emb.p")
        except:
            print("embedding to cpu error")


    numeval=10
    index=torch.ceil(torch.linspace(0,numpic-1,numeval)).to(torch.int)
    import matplotlib.pyplot as plt

    try:
        with torch.no_grad():
            print("saving colormap")
            for i in range(numeval):
                imIndex=index[i]
                currentImg=ImgTensor[imIndex,...].to(torch.float32)
                currentPose=posesTensor[imIndex,...]
                rayo,rayd=ut.getRay(currentPose,H,W,1,focal)
                rayo=rayo.permute(2,0,1)
                rayd=rayd.permute(2,0,1)
                currentImg=currentImg.permute(2,0,1)


                rayol=multigpu.datatoGPU(multigpu.splitImgToPatch(rayo,numgpu)[0],numgpu) # split on cpu
                raydl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[0],numgpu)
                splitHl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[1],numgpu)
                splitWl=multigpu.datatoGPU(multigpu.splitImgToPatch(rayd,numgpu)[2],numgpu)
                coastSigmaAllll=multigpu.datatoGPU(multigpu.splitImgToPatch(coastSigmaAlll[imIndex,...].permute(2,0,1),numgpu)[0],numgpu)
                for m in range(len(coastSigmaAllll)):
                    coastSigmaAllll[m]=coastSigmaAllll[m].permute(1,2,0)


                colormap=torch.tensor(())
                depth=torch.tensor(())
                
                for j in range(numgpu): 
                    flatPoints,flatrayd,newRayT=render.getfinepoints(coastRayTAlll[imIndex,...],sigma=coastSigmaAllll[j],rayO=rayol[j].permute(1,2,0),rayD=raydl[j].permute(1,2,0),numFine=numFinesam)

                    coastraw=render.batch_enc(enc_hash=hashcodinglist[j],enc_freq=enc_freq,flatpoints=flatPoints,flatrayd=flatrayd,fn=ml[j],if_hash=if_hash,chunkSize=chunkSize)

                    sigma,rgb=render.raw2rgb(coastraw,splitHl[j],splitWl[j],newRayT.shape[-1])

                    colormapd,depthd,accd=render.rgb2output(sigma=sigma,rgb=rgb,rayT=newRayT)
                    colormap=torch.concat((colormap,colormapd.to(torch.device("cpu"))),dim=-2)
                    depth=torch.concat((depth,depthd.to(torch.device("cpu"))),dim=-1)

                    
                colormap=colormap.detach().numpy()
                depth=depth.detach().numpy()
                currentImg=currentImg.permute(1,2,0).detach().numpy()
                plt.imsave(savepath+"color"+str(i)+".png",colormap/np.max(colormap))
                plt.imsave(savepath+"depth"+str(i)+".png",depth/np.max(depth))
                plt.imsave(savepath+"gt"+str(i)+".png",currentImg/np.max(currentImg))
        print("save pic done")
    except:
        print("save error")