import torch

import numpy as np
import os
import math
import matplotlib.pyplot as plt
def getTinyLegotensor(path):
    try:
        data=np.load(path)  
        print("read file done")
    except:
        print("read file error")    

    # %%
    images=data['images']
    poses=data['poses']
    focal=data['focal']

    # %%
    ImgTensor=torch.tensor(images)
    posesTensor=torch.tensor(poses)
    focalTensor=torch.tensor(focal)

    ImgTensor=ImgTensor[:100,:,:,:]
    posesTensor=posesTensor[:100,:,:]
    return ImgTensor,posesTensor,focalTensor



import json
import torchvision.transforms as trans
import cv2
import imageio
## basepath ~/ 
def getdatatensor(basedir, half_res=8, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    if half_res:

        H = H//half_res
        W = W//half_res
        focal = focal/float(half_res)
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    # imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
        
    return torch.tensor(imgs[...,:3]), torch.tensor(poses), torch.tensor(focal)
# def getdatatensor(basepath,ftype='train',ifdivide=True):
#     try:
#         # read img
#         datanamelist=os.listdir(basepath+ftype)
        
#         datalist=[]
#         for name in datanamelist:
#             datalist.append(plt.imread(basepath+ftype+'/'+name))
            
#         # ImgTensor=(torch.tensor(datalist)[...,:3])
#         ImgTensor=torch.tensor(datalist)
#         ImgTensor = ImgTensor[..., :3]*ImgTensor[..., -1:] + (1.-ImgTensor[..., -1:])

#         B,H,W=ImgTensor.shape[0:3]
#         ImgTensor=ImgTensor.numpy()
        
#         # read pose
#         with open(basepath+"transforms_"+ftype+".json",'r') as file:
#             file=json.load(file)
#             fullanglex=file['camera_angle_x']
#             frames=file['frames']
#             poses=[]
#             for frame in frames:
#                 poses.append(frame['transform_matrix'])
            
#             posesTensor=torch.tensor(poses)
#             focalTensor=torch.tensor(0.5*W/(np.tan(0.5*fullanglex)))
        
#         if ifdivide:
#             H = H//8
#             W = W//8
#             focalTensor = focalTensor/8.
#             imgs_half_res = np.zeros((B, H, W, 3))
#             for i, img in enumerate(ImgTensor):
#                 imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#             ImgTensor = imgs_half_res
#             # resize=trans.Resize([H,W])
#             # ImgTensor=resize(ImgTensor.permute(0,3,1,2)).permute(0,2,3,1)

#         # imgs_half_res = np.zeros((ImgTensor.shape[0], H, W, 4))
#         # for i, img in enumerate(ImgTensor):
#         #     imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         # imgs = imgs_half_res

        
#         print("read file done")
#     except:
#         print("read file error")    

    # %%
    # images=data['images']
    # poses=data['poses']
    # focal=data['focal']

    # %%
    
    # posesTensor=torch.tensor(poses)
    # focalTensor=torch.tensor(focal)

    return torch.tensor(ImgTensor),posesTensor,focalTensor