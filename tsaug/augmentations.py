__all__ = ['noise_from_random_curve', 'noise_from_normal', 'distort_time', 'AugTransform', 'YWarp', 'YNormal',
           'YNormalMul', 'YScale', 'YScaleChannel', 'TimeWarp', 'TimeNormal', 'encodes', 'encodes', 'encodes',
           'encodes', 'encodes', 'all_noise_augs', 'Zoomin', 'Zoomout', 'RandZoom', 'RandTimesteps', 'encodes',
           'encodes', 'encodes', 'encodes', 'all_zoom_augs', 'TimestepZero', 'TimestepMean', 'Cutout', 'Crop',
           'RandomCrop', 'CenterCrop', 'Maskout', 'Dimout', 'encodes', 'encodes', 'encodes', 'encodes', 'encodes',
           'encodes', 'encodes', 'encodes', 'CutoutMean', 'CenterCropMean', 'all_erasing_augs', 'all_augs',
           'RandAugment', 'Augmix', 'AugmixSS']

# import random
from functools import partial
# import pywt

import numpy as np
from scipy.interpolate import CubicSpline
# import torch


def _create_random_curve(n_channels, seq_len, magnitude, order):
    '''
    create a random curve for each channel in the interval[0, seq_len-1] on order random points
    '''
    x = np.linspace(0, seq_len-1, order)
    y = np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, len(x)))
    f = CubicSpline(x, y, axis=-1)
    return f

def noise_from_random_curve(dim, magnitude=.1, order=4):
    '''
    sample points from a gaussian with mean 1 and create a smooth cubic "random curve" from these points
    ts, needs to be 2D
    order: number of sample to create the random curve from
    '''
    n_channels, seq_len = dim
    f = _create_random_curve(n_channels, seq_len, magnitude, order)
    return torch.tensor(f(np.arange(seq_len)))

def noise_from_normal(dim, magnitude=.1):
    '''
    sample random noise from a gaussian with mean=1.0 and std=magnitude
    '''
    n_channels, seq_len = dim
    return torch.tensor(np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, seq_len)))

# Cell
def _ynoise(x, magnitude=.1, add=True, smooth=True, **kwargs):
    '''
    add random noise to timeseries values
    '''
#     assert isinstance(x, Tensor)
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    if magnitude <= 0: return x
    n_channels, seq_len = x.shape[-2], x.shape[-1]
    noise_fn = noise_from_random_curve if smooth else noise_from_normal

    noise = noise_fn((n_channels, seq_len), magnitude=magnitude, **kwargs).to(x.device)
    if add:
        output = x + (noise-1)
        return output.to(x.device, x.dtype)
    else:
        output = x * (noise)
        return output.to(x.device, x.dtype)

# Cell
_ynoise_warp = partial(_ynoise, smooth=True)
_ynoise_normal = partial(_ynoise, smooth=False)
_ynoise_normal_mul = partial(_ynoise, smooth=False, add=False)

# Cell
# smooth does not make sense probably unless there is some channel dependence and ordering
def _yscale(x, magnitude=.1, normal=False, by_channel=False):
    if magnitude <= 0: return x
    if normal:
        scale = 1.+2*(torch.randn(1 if not by_channel else x.shape[-2])-0.5)*magnitude
    else:
#         scale = 1 + torch.rand(1) * magnitude  # uniform [0:1], normal possible
        scale = 1.+2*(torch.rand(1 if not by_channel else x.shape[-2])-0.5)*magnitude
#         if np.random.rand() < .5: scale = 1 / scale # scale down
#     output = x * scale.to(x.device)
    return x*scale.to(x.device) if not by_channel else x*scale[..., None].to(x.device)

# Cell
def _normalize_timesteps(timesteps):
    '''
    distorted timesteps in [0,..,seq_len]
    '''
#     timesteps = timesteps - np.expand_dims(timesteps[:,0], -1)
#     timesteps = timesteps.clone()
    timesteps = timesteps.sub(timesteps[:,0].unsqueeze(-1))

#     timesteps = timesteps/np.expand_dims(timesteps[:,-1], -1) * (timesteps.shape[1]-1)
    timesteps=timesteps.div(timesteps[:,-1].unsqueeze(-1)) * (timesteps.shape[1]-1)

    return timesteps


def distort_time(dim, magnitude=.1, smooth=False, **kwargs):
    '''
    distort the time steps (x-axis) of timeseries
    '''
    n_channels, seq_len = dim
    noise_fn = noise_from_random_curve if smooth else noise_from_normal
    noise = noise_fn((n_channels, seq_len), magnitude=magnitude, **kwargs)
    time_new = _normalize_timesteps(noise.cumsum(1))
#     noise_cum = noise_cum - np.expand_dims(noise_cum[:,0], -1)
#     noise_cum = noise_cum/np.expand_dims(noise_cum[:,-1], -1) * (ts.shape[1]-1)
#     x /= x[-1]
#     x = np.clip(x, 0, 1)
#     print(x)
#     return x * (ts.shape[-1] - 1)
    return time_new

# Cell
def _timenoise(x, magnitude=.1, smooth=False, **kwargs):
    '''This is a slow batch tfm on cpu'''
    if magnitude <= 0: return x
#     if len(x.shape)==1: x=x.unsqueeze(0) #no support for 1D tensors
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    n_channels, seq_len = x.shape[-2], x.shape[-1]
    x_device = x.device ## make sure to put outpout on right device
    x=x.cpu() ## only works on cpu

#    return f
#     plt.plot(x.T)
#     plt.plot(np.linspace(0,10), f(np.linspace(0,10)[:, None]).squeeze())
    new_x = distort_time((n_channels,seq_len), magnitude=magnitude, smooth=True, **kwargs).to(x.device)
    fs = [CubicSpline(np.arange(seq_len), x[...,i,:], axis=-1) for i in range(n_channels)]
#     new_y = f(new_x, )
#     print(fs(new_x).shape)
#     return new_x
    new_y = torch.stack([torch.tensor(fs[i](xi)) for i,xi in enumerate(new_x)])
    if len(x.shape)==3: new_y = new_y.permute(1,0,2)

    return new_y.to(x_device, x.dtype)

# Cell
def _timewarp(x, magnitude=.1, order=4):
    return _timenoise(x, magnitude, smooth=True, order=order)
def _timenormal(x, magnitude=.1):
    return _timenoise(x, magnitude, smooth=False)

# Cell
def _randomize(p):
    p = np.random.beta(p,p)
    return np.maximum(p, 1-p)

# Cell
def _rand_steps(n, p, rand=False, window=False):
    if rand: p = _randomize(p)
    n_steps = int(p*n)
    if window:
        start = np.random.randint(0, n-n_steps+1)
        return np.arange(start, start+n_steps)
    else: return np.sort(np.random.choice(n, n_steps, replace=False))

# Cell
def _zoom(x, magnitude=.2, rand=False, zoomout=False, window=True, verbose=False):
    '''This is a slow batch tfm
    win_len: zoom into original ts into a section consisting of win_len original data points
    randomly choose one of the seq_len-win_len possible starting points for that section
    within that section, consider seq_len(number of original datapoints) evenly distributed new datapoints
    and interpolate the respective values with a cubic spline
    '''
    if magnitude == 0: return x
#     x_device = x.device ## make sure to put outpout on right device
#     x=x.cpu() ## only on cpu with CubicSpline

    n_channels, seq_len = x.shape[-2], x.shape[-1]
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'

    window=_rand_steps(seq_len, 1-magnitude, rand=rand, window=window)
    if zoomout: window=np.arange(seq_len-len(window), seq_len)
    pv(window, verbose)
#     x2 = x[..., window]
    fs = [CubicSpline(np.arange(len(window)), x[...,i, window].cpu(), axis=-1) for i in range(n_channels)]
    output = torch.stack(
        [torch.tensor(fs[i](np.linspace(0,len(window)-1, num=seq_len))) for i in range(n_channels)])
    if len(x.shape)==3: output = output.permute(1,0,2)
#     output = x.new(f(np.linspace(0, len(window) - 1, num=seq_len)))

#     new_y = torch.stack([torch.tensor(fs[i](xi)) for i,xi in enumerate(new_x)])
#     if len(x.shape)==3: new_y = new_y.permute(1,0,2)

#     return new_y.to(x_device, x.dtype)


    return output.to(x.device, x.dtype)


# Cell
_zoomin = partial(_zoom, rand=True)
_zoomout = partial(_zoom, rand=True, zoomout=True)

def _randzoom(x, magnitude=.2):
    p = np.random.rand()
    return _zoomin(x, magnitude) if p<0.5 else _zoomout(x, magnitude)

# Cell
_randtimesteps = partial(_zoom, window=False)

# Cell
def _complement_steps(n, steps, verbose=False):
    pv('complement', verbose)
    pv(n, verbose)
    pv(steps, verbose)
    return np.sort(np.array(list(set(n)-set(steps))))

# Cell
def _center_steps(n, steps):
    start = n//2-len(steps)//2
    return np.arange(start, start+len(steps))

# Cell
def _create_mask_from_steps(x, steps, dim=False):
    '''create a 2D mask'''
    mask = torch.zeros_like(x, dtype=torch.bool)
#     print(mask.shape)
#     print(steps)
#     print(mask[steps,:])
#     print(mask[:, steps])
    if dim:
        mask[steps, :] = True
    else:
        mask[:, steps] = True
    return mask

# Cell
def _erase(x, magnitude=.2, rand=False, window=False, mean=False, complement=False, center=False, mask=False,
          dim=False, verbose=False):
    '''erasing parts of the timeseries'''
    if magnitude==0: return x

    pv(f'_erase input shape {x.shape}', verbose)
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    is_batch = len(x.shape)==3

    pv(x.shape, verbose)

    n_channels, seq_len = x.shape[-2], x.shape[-1]
    p = 1-magnitude if complement else magnitude

    n = n_channels if dim else seq_len
    steps = _rand_steps(n, p, rand=rand, window=window)
    if center: steps = _center_steps(n, steps)
    if complement: steps = _complement_steps(np.arange(n), steps, verbose=verbose)

    pv(f'steps {steps}', verbose)
    output = x.clone()
    if not is_batch: output.unsqueeze_(0)
    value = 0 if not mean else output.mean((0,2), keepdims=True)
    mask = torch.rand_like(output[0])<magnitude if mask else _create_mask_from_steps(output[0], steps, dim=dim)

    pv(mask, verbose)
    pv(value, verbose)

    if not mean: output[..., mask] = 0
    else:
        assert mask.shape[-2] == value.shape[-2]
        output[..., mask]=0
        output.add_(mask.int().to(x.dtype).unsqueeze(0)*value)
    return output.squeeze_(0) if not is_batch else output


# Cell
_timestepzero = partial(_erase)
_timestepmean = partial(_erase, mean=True)
_cutout = partial(_erase, rand=True, window=True, complement=True)
_crop = partial(_erase, window=True, complement=True)
_randomcrop = partial(_erase, window=True, rand=True, complement=True)
_centercrop = partial(_erase, window=True, rand=True, center=True,complement=True)
_maskout = partial(_erase, mask=True)
_dimout = partial(_erase, dim=True)

# Cell
@delegates(to=Transform.__init__) ## delegates should default to base class
class AugTransform(Transform):
    split_idx,init_enc,order,train_setup = 0,True,90,None
    def __init__(self, magnitude=0.1, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.magnitude=magnitude
        self.verbose = verbose

    def __call__(self, x, split_idx=split_idx):
        return super().__call__(x, split_idx=split_idx)

#     def encodes(self, x:TSIntTensor):
#         return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

# Cell
# @delegates(keep=True)
class YWarp(AugTransform):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

    def encodes(self, x:TSTensor):
        pv('ywarp', self.verbose)
        return _ynoise_warp(x, magnitude=self.magnitude)

# Cell
class YNormal(AugTransform):
    def encodes(self, x:TSTensor):
        pv('ynormal', verbose=self.verbose)
        return _ynoise_normal(x)

class YNormalMul(AugTransform):
    def encodes(self, x:TSTensor):
        pv('ynormal', verbose=self.verbose)
        return _ynoise_normal_mul(x)

# Cell
class YScale(AugTransform):
    def encodes(self, x:TSTensor):
        pv('yscale', verbose=self.verbose)
        return _yscale(x)

# Cell
class YScaleChannel(AugTransform):
    def encodes(self, x:TSTensor):
        pv('yscale channel', verbose=self.verbose)
        return partial(_yscale, by_channel=True)(x)

# Cell
class TimeWarp(AugTransform):
    def encodes(self, x:TSTensor):
        pv('timewarp', verbose=self.verbose)
        return _timewarp(x)

# Cell
class TimeNormal(AugTransform):
    def encodes(self, x:TSTensor):
        pv(f'timenormal {x.shape}', verbose=self.verbose)
        return _timenormal(x, magnitude=self.magnitude)

# Cell
@YWarp
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
#     return cast(self.__call__(TSTensor(x.float())).int(), type(x))

@YNormal
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
#     return cast(self.__call__(TSTensor(x.float())).int(), type(x))

@YScale
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

@YScaleChannel
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

@TimeWarp
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))


# Cell
def all_noise_augs(magnitude=0.1, verbose=False):
    kwargs = {'magnitude':magnitude, 'verbose':verbose}
    return [YWarp(**kwargs), YNormal(**kwargs), YScale(**kwargs), YScaleChannel(**kwargs),
           TimeWarp(**kwargs), TimeNormal(**kwargs)]

# Cell
class Zoomin(AugTransform):
    def encodes(self, x:TSTensor):
        pv('zoomin', verbose=self.verbose)
        return _zoomin(x, magnitude=self.magnitude)

class Zoomout(AugTransform):
    def encodes(self, x:TSTensor):
        pv('zoomout', verbose=self.verbose)
        return _zoomout(x, magnitude=self.magnitude)

class RandZoom(AugTransform):
    def encodes(self, x:TSTensor):
        pv('randzoom', verbose=self.verbose)
        return _randzoom(x, magnitude=self.magnitude)

class RandTimesteps(AugTransform):
    def encodes(self, x:TSTensor):
        pv('randtimestep', verbose=self.verbose)
        return _randtimesteps(x, magnitude=self.magnitude)

# Cell
@Zoomin
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

@Zoomout
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

@RandZoom
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

@RandTimesteps
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))


# Cell
def all_zoom_augs(magnitude=0.1, verbose=False):
    kwargs = {'magnitude':magnitude, 'verbose':verbose}
    return [Zoomin(**kwargs), Zoomout(**kwargs), RandZoom(**kwargs),
           RandTimesteps(**kwargs)]

# Cell
class TimestepZero(AugTransform):
    def encodes(self, x:TSTensor):
        pv('timestepzero', verbose=self.verbose)
        return _timestepzero(x, magnitude=self.magnitude)

class TimestepMean(AugTransform):
    def encodes(self, x:TSTensor):
        pv('timestepmean', verbose=self.verbose)
        return _timestepmean(x, magnitude=self.magnitude)

class Cutout(AugTransform):
    def encodes(self, x:TSTensor):
        pv('cutout', verbose=self.verbose)
        return _cutout(x, magnitude=self.magnitude)

class Crop(AugTransform):
    def encodes(self, x:TSTensor):
        pv('crop', verbose=self.verbose)
        return _crop(x, magnitude=self.magnitude)

class RandomCrop(AugTransform):
    def encodes(self, x:TSTensor):
        pv('randomcrop', verbose=self.verbose)
        return _randomcrop(x, magnitude=self.magnitude)

class CenterCrop(AugTransform):
    def encodes(self, x:TSTensor):
        pv('centercrop', verbose=self.verbose)
        return _centercrop(x, magnitude=self.magnitude)

class Maskout(AugTransform):
    def encodes(self, x:TSTensor):
        pv('maskout', verbose=self.verbose)
        return _maskout(x, magnitude=self.magnitude)

class Dimout(AugTransform):
    def encodes(self, x:TSTensor):
        pv('dimout', verbose=self.verbose)
        return _dimout(x, magnitude=self.magnitude)

# Cell
@TimestepZero
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@TimestepMean
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@Cutout
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@Crop
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@RandomCrop
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@CenterCrop
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@Maskout
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
@Dimout
def encodes(self, x:TSIntTensor):
    return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))


# Cell
class CutoutMean(AugTransform):
    def encodes(self, x:TSTensor):
        pv('cutout', verbose=self.verbose)
        return partail(_cutout, mean=True)(x, magnitude=self.magnitude)
    def encodes(self, x:TSIntTensor):
        return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

class CenterCropMean(AugTransform):
    def encodes(self, x:TSTensor):
        pv('centercrop', verbose=self.verbose)
        return partial(_centercrop, mean=True)(x, magnitude=self.magnitude)
    def encodes(self, x:TSIntTensor):
        return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))


# Cell
def all_erasing_augs(magnitude=0.1, verbose=False):
    kwargs = {'magnitude':magnitude, 'verbose':verbose}
    return [Dimout(**kwargs), Cutout(**kwargs), TimestepZero(**kwargs), Crop(**kwargs),
            RandomCrop(**kwargs), Maskout(**kwargs)]

# Cell
def all_augs(magnitude=0.3, verbose=False):
    kwargs = {'magnitude':magnitude, 'verbose':verbose}
    return all_noise_augs(**kwargs)+all_zoom_augs(**kwargs)+all_erasing_augs(**kwargs)

# Cell
class RandAugment(AugTransform):
    '''
    choose `N` transforms randomly and apply them with magnitude `magnitude`
    '''
    def __init__(self, N=3, magnitude=0.3, tfms=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        pv(f'tfms {tfms}', verbose)
        self.tfms = tfms
        if tfms is None: self.tfms = all_augs(magnitude, verbose=verbose)

    def encodes(self, x:TSTensor):
        fs = np.random.choice(self.tfms, self.N, replace=False)
        return compose_tfms(x, fs)

    def encodes(self, x:TSIntTensor):
        return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

# Cell
class Augmix(AugTransform):
    def __init__(self, N=3, magnitude=0.2, k=3, alpha=1, tfms=None, verbose=False, **kwargs):
        '''
        N: number of operations per update
        k: number of update operations
        alpha: beta and dirichlet parameter
        '''
        super().__init__(**kwargs)
        self.N, self.k, self.alpha, self.verbose = N, k, alpha, verbose
#         print(f'tfms {tfms}')
        self.tfms = tfms
        if tfms is None: self.tfms = all_augs(magnitude, verbose=verbose)




    def encodes(self, x:TSTensor):
        x_aug = torch.zeros_like(x)
        weights = np.random.dirichlet([self.alpha]*self.k)
        m = np.random.beta(self.alpha, self.alpha)
        pv(f'weights {weights}, m {m}', self.verbose)
        for i in range(self.k):
            fs = np.random.choice(self.tfms, self.N, replace=False)
            n = np.random.randint(self.N)
            pv(f'ops in step {i} {fs[:n+1]}', self.verbose)
            x_aug += weights[i]*compose_tfms(x, fs[:n+1]) ## normalize?
        return m*x + (1-m)*x_aug

    def encodes(self, x:TSIntTensor):
        return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))

# Cell
class AugmixSS(AugTransform):
    '''
    augmix with standardization
    '''
    def __init__(self, N=3, magnitude=0.3, k=3, alpha=1, tfms=None, verbose=False,
                 mean=None, std=None, **kwargs):
        '''
        N: number of operations per update
        k: number of update operations
        alpha: beta and dirichlet parameter
        '''
        super().__init__(**kwargs)

        self.mean = tensor(mean) if mean is not None else None
        self.std = tensor(std) if std is not None else None

        self.N, self.k, self.alpha, self.verbose = N, k, alpha, verbose
#         print(f'tfms {tfms}')
        self.tfms = tfms
        if tfms is None: self.tfms = all_augs(magnitude, verbose=verbose)
        self.axes=(0,2) #hard coded for now


    def setups(self, dl: DataLoader):
        if self.mean is None or self.std is None:
            #pv(f'{self.__class__.__name__} setup mean={self.mean}, std={self.std}, by_sample={self.by_sample}, by_var={self.by_var}', self.verbose)
#             x, *_ = dl.one_batch() ##??
            x = dl.ptls[0] ## modification
            self.mean = x.mean(self.axes, keepdim=self.axes!=()).to(default_device())
            self.std =  x.std(self.axes, keepdim=self.axes!=()).to(default_device()) + 1e-7
            pv(f'mean: {self.mean}  std: {self.std}\n', self.verbose)

    def _normalize(self,x):
        return (x-self.mean)/self.std


    def encodes(self, x:TSTensor):
        x_aug = torch.zeros_like(x)
        weights = np.random.dirichlet([self.alpha]*self.k)
        m = np.random.beta(self.alpha, self.alpha)
        pv(f'weights {weights}, m {m}', self.verbose)
        for i in range(self.k):
            fs = np.random.choice(self.tfms, self.N, replace=False)
            n = np.random.randint(self.N)
            pv(f'ops in step {i} {fs[:n+1]}', self.verbose)
            x_aug += weights[i]*self._normalize(compose_tfms(x, fs[:n+1])) ## normalize?
        return m*self._normalize(x) + (1-m)*x_aug

#     def encodes(self, x:TSIntTensor):
#         return cast(self.__call__(TSTensor(x.float())).type(x.dtype), type(x))
