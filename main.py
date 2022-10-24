import train
import model as NeRFmodel
import torch
import eval
import dataset as td
import configargparse
import os
import time
import encodings as enc

from utils.encoding import FreqEncoding

currentdir=os.getcwd()

parser = configargparse.ArgumentParser()
parser.add_argument("--basedir", type=str, default=currentdir,
                        help='where to store ckpts and logs')
parser.add_argument("--savedir", type=str, default=currentdir+'/experiments/1/',
                        help='where to store results')
parser.add_argument("--tinydatadir", type=str, default=currentdir+'/data/train_data/tiny_nerf_data.npz',
                        help='where load tiny lego data')
parser.add_argument("--legodatadir", type=str, default=currentdir+'/data/train_data/lego/',
                        help='where load full lego data')
parser.add_argument("--ifhash", type=int, default=0,
                        help='if use hash encoding')     
parser.add_argument("--numgpu", type=int, default=1,
                        help='numgpus')     
# if use tiny lega dataset   default:1
parser.add_argument("--iftiny", type=int, default=1,
                        help='if use tiny lega dataset')  
parser.add_argument("--numepoch", type=int, default=1000,
                        help='numepoch') 
parser.add_argument("--numcoast", type=int, default=64,
                        help='num coast')  
parser.add_argument("--numfine", type=int, default=64,
                        help='num fine')  
parser.add_argument("--chunksize", type=int, default=1024*32,
                        help='num chunk size')    
parser.add_argument("--nrand", type=int, default=200*200,
                        help='num random patch')    
parser.add_argument("--resscale", type=int, default=8,
                        help='resize scale')                                  
args = parser.parse_args()




if __name__=='__main__':
    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)



    if args.iftiny:
        ImgTensor,posesTensor,focalTensor=td.getTinyLegotensor(args.tinydatadir)
        H,W=ImgTensor.shape[1:3]
    else:
        ImgTensor,posesTensor,focalTensor=td.getdatatensor(args.legodatadir,half_res=args.resscale)
        H,W=ImgTensor.shape[1:3]


    if args.ifhash:
        nerfmodel=NeRFmodel.NeRFtinymodel()
        embeddings = torch.nn.ModuleList([torch.nn.Embedding(2**19, 2) for i in range(16)])
        enc_freq=FreqEncoding()
    else:
        nerfmodel=NeRFmodel.NeRFmodel()
        embeddings = ([None for i in range(16)])
        enc_freq=FreqEncoding()

    start=time.time()
    ml,emb_list,coastRayTAlll,coastSigmaAlll,hashcodinglist=train.train(ImgTensor,posesTensor,focalTensor,model=nerfmodel,if_hash=args.ifhash,numgpu=args.numgpu,enc_hash=embeddings,enc_freq=enc_freq,iter_num=args.numepoch,numCoastsam=args.numcoast,numFinesam=args.numfine,chunkSize=args.chunksize,N_rand=args.nrand)
    end=time.time()
    print("train time="+str(start-end))

    
    start=time.time()
    eval.eval_nerf_fine(model=nerfmodel,ml=ml,emb=embeddings,emb_list=emb_list,hashcodinglist=hashcodinglist,ImgTensor=ImgTensor,posesTensor=posesTensor,
                        focal=focalTensor,coastRayTAlll=coastRayTAlll,coastSigmaAlll=coastSigmaAlll,numgpu=args.numgpu,if_hash=args.ifhash,savepath=args.savedir,numFinesam=args.numfine,chunkSize=args.chunksize,enc_freq=enc_freq)
    end=time.time()
    print("eval time="+str(start-end))
