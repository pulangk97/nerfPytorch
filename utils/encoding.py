from cmath import log
import torch

import numpy as np

import math

import utils.utils as ut


## frequency encoding 
## input x [B,3] position tensor , l num of embs
## output 
class FreqEncoding(torch.nn.Module):
        def __init__(self,enc_l=6):
            super().__init__()
            self.enc_l=enc_l
           
        def posEnc(self,x,l):
            gama=[x]
            for i in range(l):           
                    gama.append(torch.cos((2**i)*torch.pi*x))
                    gama.append(torch.sin((2**i)*torch.pi*x)) 
            return torch.concat(gama,-1)

        def forward(self,input):
            return self.posEnc(input,self.enc_l)



class HashEncoding(torch.nn.Module):
    def __init__(self, bounding_box, embedding_input,n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        self.rundevice=bounding_box.device
        self.finest_resolution=torch.tensor(finest_resolution,device=self.rundevice)
        self.base_resolution=torch.tensor(base_resolution,device=self.rundevice)
        self.n_levels=torch.tensor(n_levels,device=self.rundevice)
        self.bounding_box=bounding_box
        self.n_features_per_level=n_features_per_level
        self.log2_hashmap_size=log2_hashmap_size
        self.b=torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        self.resolutions=[torch.floor(self.base_resolution*self.b**i)  for i in range(self.n_levels)]
        self.embeddings =embedding_input
        # self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(2**self.log2_hashmap_size, \
        #                                 self.n_features_per_level,device=self.rundevice) for i in range(n_levels)])



    ## points [N,3] tensor bounding_box [2,3] numpy or tensor
    def getvoxelembedding(self,points,i_levels,bounding_box):
        step=torch.concat([(bounding_box[1,i,np.newaxis]-bounding_box[0,i,np.newaxis])/self.resolutions[i_levels]  for i in range(3)],dim=-1)
        # try:
        #         step=torch.tensor([(bounding_box[1,i]-bounding_box[0,i])/self.resolutions[i_levels].numpy()  for i in range(3)])

        # except:
        #         bounding_box=bounding_box.numpy()
        #         step=torch.tensor([(bounding_box[1,i]-bounding_box[0,i])/self.resolutions[i_levels].numpy()  for i in range(3)])

        ## delete all points on boundary
        ## 
        # points=points[list((points[...,0].numpy()>bounding_box[0,0].numpy())&(points[...,0].numpy()<bounding_box[1,0].numpy())),:]
        # points=points[list((points[...,1].numpy()>bounding_box[0,1].numpy())&(points[...,1].numpy()<bounding_box[1,1].numpy())),:]
        # points=points[list((points[...,2].numpy()>bounding_box[0,2].numpy())&(points[...,2].numpy()<bounding_box[1,2].numpy())),:]


        ## true position to hashmap
        ##
        # bound=[[torch.floor((points[...,i]-bounding_box[0,i])/step[i])*step[i]+bounding_box[0,i],torch.floor((points[...,i]-bounding_box[0,i])/step[i])*step[i]+bounding_box[0,i]+step[i]]  for i in range(3)],
        # pi=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        # p=[]
        # for i in range(8):
        #         p.append([bound[j][pi[i][j]].numpy() for j in range(3)])
        # p=torch.tensor(p).permute(2,0,1)
        # pindis=self.hash(p,self.log2_hashmap_size)
        # emb=self.embeddings[i_levels](pindis)
        # return emb
        ##
        ##

        lowerbound=torch.floor((points-bounding_box[0,np.newaxis,...])/step[np.newaxis,:]).to(torch.int)
        pi=torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype=torch.int,device=self.rundevice)
        pindis=pi[np.newaxis,...]+lowerbound[:,np.newaxis,:]
        hashdis=self.hash(pindis,self.log2_hashmap_size)

        emb=self.embeddings[i_levels](hashdis)
        floor_bound=lowerbound*step[np.newaxis,:]+bounding_box[0,np.newaxis,...]
        ceil_bound=(lowerbound+pi[7,np.newaxis,...])*step[np.newaxis,:]+bounding_box[0,np.newaxis,...]

        return emb ,floor_bound , ceil_bound


        
    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def hash(self,coords, log2_hashmap_size):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords,device=self.rundevice)[..., 0]
        for i in range(coords.shape[-1]):
                xor_result ^= coords[..., i]*primes[i]

        return torch.tensor((1<<log2_hashmap_size)-1,device=self.rundevice).to(xor_result.device) & xor_result   


    def forward(self,flatpoints):
        emb_all=[]
        for i in range(self.n_levels):
                emb,lb,cb=self.getvoxelembedding(flatpoints,i,self.bounding_box)  
                emb_all.append(self.trilinear_interp(flatpoints,lb,cb,emb))

        return torch.cat(emb_all, dim=-1)
        
