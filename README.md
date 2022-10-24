# nerfPytorch (multi GPU)
NeRF realize by Pytorch  
## content  
1. noval 'lego' view synthesis  
2. positional encoding by hash ecoding or freq encoding  
3. multi gpu realize  

## how to start?  
1. Download lego data from [datasets](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)  
2. put 'lego' data directory into data/train_data/ directory  
3. start command  
```
python main.py
```
4. parameters
```
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

```

