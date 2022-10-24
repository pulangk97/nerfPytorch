# nerfPytorch
NeRF realize by Pytorch  
## content  
1. noval 'lego' view synthesis  
2. positional encoding by hash ecoding or freq coding  
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
# NeRF  
## introduction
> [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf)

NerF作为一种隐式神经表达三维场景的方式，在多视角二维图像监督下，渲染出三维场景，并且渲染结果是连续的，能够隐式将(x,y,z,**d**)坐标映射为($\sigma$,**c**)。$$f:(x,y,z,\vec{d})\Rightarrow(\sigma,\vec{c})\tag{1}$$其中**c**为RGB编码(**可以看作是散射光谱信息**)，仅与散射/漫反射物体的坐标(x,y,z)以及辐射方向**d**有关，也就是仅与观测物体的**物质特性、表面结构以及辐射方向***有关，实际上RGB光谱信息随着照明光入射方向同样有关，由于NeRF在LDR场景下采集图像，可以看作照明光强和照明方向是均匀的(**在真实场景下可能需要对照明光场进行建模并且光谱数据需要表示为随照明方向变化的函数**)。  **c**并非物体本征特性，而是表示的是在一定照明条件下（均匀照明）散射光谱和强度变化分布，相当于在体密度表征的辐射强度基础上加入光谱、viewdependent信息，利用不同视角的c能够在照明已知条件下估计物体本征材质以及纹理。所以c重建结果实际上是物体本征材质和纹理与光照作用后的结果。
体密度$\sigma$作为一种物体本征特性的抽象，**实际上不仅仅代表密度这个物理量，而是作为影响辐射率(散射率或反射率，单位光子数照明下向外辐射的光子数)的一个抽象的参数**。某一点的辐射率看作是体密度和体积的求积概率表示$1-exp(-\sigma\delta)$，体密度的增加会带来辐射率的增加，同时剩余光子数（辐射方向不发生变化）表示为$exp(-\sigma\delta)$，这里认为光与物质的作用仅仅是散射或漫反射消光，*实际上物质还存在吸收消光*。
NeRF模型对观测物体实际上做了一定的假设，认为观测物体的材质固定且是各向同性的漫反射物体，并且为了恢复体密度光谱这种物质本征特性还需要均匀的照明条件，这在真实情况下难以做到，虽然可以通过相机后处理过程实现但是这种方法破坏了原始图像噪声分布从而导致渲染颜色发生变化([NeRF in the Dark](https://bmild.github.io/rawnerf/))。  
因此后续针对这个问题涌现许多工作，对后续网络结构以及建模过程进行优化，实现对不同材质反射结构的区分。
## Method
### 成像模型
![NeRF](../nerf+holography/nerfpic/1.png)
整个成像模型可以表达为式(2)$$C(\mathbf{r})=\int_{t_{n}}^{t_{f}} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) d t, \text { where } T(t)=\exp \left(-\int_{t_{n}}^{t} \sigma(\mathbf{r}(s)) d s\right) \tag{2} $$ 成像模型看作是相机像素在物空间投影区域的辐射到达该像素的积分，实际上每个像素的物空间等效为一条光线(实际上这种在理想直线上采样的方式会导致高频信息混叠，后续[Mip-NeRF](https://jonbarron.info/mipnerf/)的工作则针对这一问题将物空间等效为三维高斯分布)。
### 数据准备
监督数据为对同一场景拍摄多幅不同视角的图像(保证照明均匀)，将多视角图像输入到comcol软件中计算相机pose，pose矩阵维度为4*4，代表从相机坐标系到全局坐标系的空间变换。其中前三行三列代表旋转矩阵，第4列前三行代表平移量，空间坐标变换看作旋转+平移。$$\left[\begin{array}{c}
x_{c} \\
y_{c} \\
z_{c} \\
1
\end{array}\right]=\left[\begin{array}{cccc}
n_{x} & o_{x} & a_{x} & t_{x} \\
n_{y} & o_{y} & a_{y} & t_{y} \\
n_{z} & o_{z} & a_{z} & t_{z} \\
0 & 0 & 0 & 1
\end{array}\right]\left[\begin{array}{c}
x_{w} \\
y_{w} \\
z_{w} \\
1
\end{array}\right]\tag{3}$$

根据相机坐标系，相机高宽，以及小孔成像等效的焦距，能够计算出每个像素对应的采样光线矢量的原点位置$\vec o$以及光线方向$\vec d$。
```
def getRay(pose,H,W,pixSize,Focus):
    xs = torch.linspace(0, W, steps=W,dtype=torch.float32)
    ys = torch.linspace(0, H, steps=H,dtype=torch.float32)
    i, j = torch.meshgrid(xs, ys, indexing='xy')

    dirs = torch.stack([(i-W*.5)/Focus, -(j-H*.5)/Focus, -torch.ones_like(i)], -1)
    # print(i)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3,:3], -1)
    rays_o = torch.broadcast_to(pose[:3,-1], rays_d.shape)
    return rays_o, rays_d
```
在近焦到远焦距离的均匀分层中每层随机采样一个点到光线原点的距离t，计算出该点空间位置$\vec o+t\vec d$，之后对位置以及光线方向进行位置编码。$$\gamma(p)=\left(\sin \left(2^{0} \pi p\right), \cos \left(2^{0} \pi p\right), \cdots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$$  
```
def posEnc(x,l):
    gama=[x]
    for i in range(l):           
            gama.append(torch.cos((2**i)*torch.pi*x))
            gama.append(torch.sin((2**i)*torch.pi*x)) 
    return torch.concat(gama,-1)
```


### 渲染

根据式（2）对



# NeRF with reflections 
> [NeRFReN: Neural Radiance Fields with Reflections](https://bennyguo.github.io/nerfren/.)  

**原始的NeRF针对各向同性的漫反射/散射进行建模，对于镜面反射并没有有效建模，实际上对于镜面反射辐射强度是各向异性的，模型与漫反射和散射有很大差异，导致NeRF对镜面反射区域深度估计不准确**(原始的NeRF建模方式重建出结果认为镜中依然有物体，估计的深度是被反射物体的距离)，更严重的问题是，会造成镜面后方物体渲染的模糊(渲染时认为镜面成像结果位于镜面后方，造成后方物体渲染误差)<br>
解决方案：**对镜面成像进行建模**
认为图像为漫反射成像$I_t$与镜面反射成像$I_r$叠加形成，这需要对两部分建模分别进行优化。$$I=I_t+ \beta I_r \tag{1}$$ 
![NeRF](../nerf+holography/nerfpic/2.png)
原始的MLP结构被分为两部分，一部分用于预测实际物体的体密度，反射率以及颜色，另一部分用于预测反射虚像的体密度和颜色。$$\widehat{\mathbf{C}}=\widehat{\mathbf{C}}\left(\mathbf{r} ; \sigma^{t}, \mathbf{c}^{t}\right)+\beta\left(\mathbf{r} ; \sigma^{t}, \alpha\right) \widehat{\mathbf{C}}\left(\mathbf{r} ; \sigma^{r}, \mathbf{c}^{r}\right)\tag{2}$$ 最终渲染的颜色看作是漫反射渲染和镜面反射渲染之和，其中镜面反射渲染受到反射系数$\beta$的加权.$$\beta(\mathbf{r} ; \sigma, \alpha)=\sum_{k} T_{i}\left(\sigma^{t}\right)\left(1-\exp \left(-\sigma_{i}^{t} \delta_{i}\right)\right) \alpha_{i} \tag{3}$$<br>将场景分解为传输分量和反射分量是一个约束不足的问题。有无数的解和糟糕的局部极小值可能产生视觉上令人满意的渲染结果，但不能分离反射辐射场和透射辐射场。人类能正确识别反射的虚拟图像，因为我们知道真实世界的几何形状,引入深度平滑先验以及双向深度一致性先验能够有效对建模优化结果进行约束，使得能够求解出镜面区域。
## 深度平滑先验
在镜面反射区域，透射体密度估计的深度应当是足够平滑的，如果优化过程在某些区域出现深度不平滑变化的情况，比如虚像和实像重合，深度估计出现误差，那么这个先验约束会让优化求解趋近于在该区域建立反射成像模型。$$\begin{aligned} \mathcal{L}_{d}=& \sum_{p} \sum_{q \in \mathcal{N}(p)} \omega(p, q)\left\|t^{*}(p)-t^{*}(q)\right\|_{1} \\ & \omega(p, q)=\exp \left(-\gamma\|\mathbf{C}(p)-\mathbf{C}(q)\|_{1}\right) \end{aligned}\tag{4}$$ 其中$t(p)$ 为深度计算式 ，$t^{*}(\mathbf{r} ; \sigma)=\sum_{k} \omega_{i} t_{i}=\sum_{k} T_{i}(\sigma)\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) t_{i}$ ，实际上就是找到权重最大的表面处对应的深度。 
## 双向深度一致性先验
同时在镜面反射区域，光线前后估计深度值应当是一致的，如果出现不一致的情况，则是优化过程中存在反射虚像的深度估计结果与实像的估计结果重合，此时估计深度会存在较大误差，因此双向深度一致性先验趋向于对这种情况逐渐约束为两种组分的叠加形式。$$\mathcal{L}_{b d c}=\sum_{\mathbf{r}}\left\|\vec{t}^{*}\left(\mathbf{r} ; \sigma^{r}\right)-\overleftarrow{t} *\left(\mathbf{r} ; \sigma^{r}\right)\right\|_{1}\tag{5}$$ 其中$\overleftarrow{t}^{*}(\mathbf{r} ; \sigma)=\sum_{k} \overleftarrow{\omega}_{i} t_{i}=\sum_{k} \overleftarrow{T}_{i}(\sigma)\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) t_{i}$。
## Loss Function
loss为重建loss，深度平滑loss，双向深度一致loss加权组合而成，为了避免整体学习趋近于学习后两项导致局部最小，需要采用warm-up策略，先令权重足够小，使得先能够有效渲染场景，之后再逐渐增大减小权重逐渐加入反射模型。
$$\mathcal{L}=\mathcal{L}_{p m}+\lambda_{d} \mathcal{L}_{d}+\lambda_{b d c} \mathcal{L}_{b d c}\tag{6}$$  
## 无纹理反射
这种情况实际上没有实像和虚像的混叠，采用上述两种先验loss没有办法有效区分是否是镜面反射，需要人工进行标记，并对反射系数和标记做loss，令该区域建模为反射模型$$\mathcal{L}_{\beta}=\sum_{p}\|\widehat{\beta}(p)-\beta(p)\|_{1}\tag{7}$$
## results
最终结果能够分别重建出虚像模型和实像模型，在加入两个先验之后实像模型的深度估计更加准确。
![NeRF](../nerf+holography/nerfpic/3.png)
无纹理反射需要人工标记反射区域，完成反射建模。
![NeRF](../nerf+holography/nerfpic/4.png)
对比实验
![NeRF](../nerf+holography/nerfpic/5.png)

