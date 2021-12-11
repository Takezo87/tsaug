from typing import Optional, List, Tuple, Any
from functools import partial

import numpy as np
from scipy.interpolate import CubicSpline

from .utils import noise_from_normal, noise_from_random_curve, integer_noise_from_normal

####################
#functional timeseries augmentations:
#    accepted input is a 2D or 3D numpy array
####################

### y noise

def _ynoise(x: np.ndarray, magnitude: float = .1, add: bool =True, smooth: bool =True, **kwargs):
    '''
    add random noise to timeseries values
    '''
#     assert isinstance(x, Tensor)
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    if magnitude <= 0: return x
    n_channels, seq_len = x.shape[-2], x.shape[-1]
    noise_fn = noise_from_random_curve if smooth else noise_from_normal

    # noise = noise_fn((n_channels, seq_len), magnitude=magnitude, **kwargs).to(x.device)
    noise = noise_fn((n_channels, seq_len), magnitude=magnitude, **kwargs)
    if add:
        output = x + (noise-1)
        return output
        # return output.to(x.device, x.dtype)
    else:
        output = x * (noise)
        return output
        # return output.to(x.device, x.dtype)


def ynoise_normal_add(x: np.ndarray, magnitude: float = .1):
    """
    add additive noise to timeseries x
    noise taken from normal distribution with mean 0. and stadard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of noise distribution

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _ynoise(x, magnitude=magnitude, add=True, smooth=False)

def _logit(x: np.ndarray) -> np.ndarray:
    assert (0<=x).all() and (x<=1).all(), '_logit expects arrays with values in [0,1]'
    return np.log(x/(1-x))

def to_logit(x: np.ndarray, format:str='decimal') -> np.ndarray:
    """
    transform x into logits
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        format: either 'decimal' (decimal odds) or 'probability'

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    output = 1./x if format=='decimal' else x
    return _logit(output)

def odds_noise(x: np.ndarray, magnitude: float = .1) -> np.ndarray:
    """
    add additive noise to a probability timeseries x
    noise taken from normal distribution with mean 0. and stadard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of noise distribution

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    output = _logit(x)
    return _ynoise(output, magnitude=magnitude, add=True, smooth=False)

def ynoise_normal_mul(x: np.ndarray, magnitude: float = .1):
    """
    add multiplicative noise to timeseries x
    noise taken from normal distribution with mean 1. and stadard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of noise distribution

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _ynoise(x, magnitude=magnitude, add=False, smooth=False)

def ynoise_normal_warp(x: np.ndarray, magnitude: float = .1):
    """
    add additive noise to timeseries x
    noise taken from a smooth random curve fitted on random poiunts sampled from
    a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _ynoise(x, magnitude=magnitude, add=True, smooth=True)

# _ynoise_warp = partial(_ynoise, smooth=True)
# _ynoise_normal = partial(_ynoise, smooth=False)
# _ynoise_normal_mul = partial(_ynoise, smooth=False, add=False)

### y scaling

def _yscale(x, magnitude=.1, normal=False, by_channel=False):
    '''
    y-scaling, random scaling factor from normal or uniform distribution, per channel possible
    '''
    if magnitude <= 0: return x
    if normal:
        #magnitude: stdev
        scale = 1.+(np.random.randn(1 if not by_channel else x.shape[-2]))*magnitude
    else:
        #uniformly from [1-magnitude, 1+magnitude]
#         scale = 1 + torch.rand(1) * magnitude  # uniform [0:1], normal possible
        scale = 1.-magnitude+2*(np.random.rand(1 if not by_channel else x.shape[-2]))*magnitude
#         if np.random.rand() < .5: scale = 1 / scale # scale down
#     output = x * scale.to(x.device)
    # print(scale)
    return x*scale if not by_channel else x*scale[..., None]
    # return x*scale.to(x.device) if not by_channel else x*scale[..., None].to(x.device)


def yscale_normal(x: np.ndarray, magnitude: float = .1):
    """
    "
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    from a normal distribtution with mean 0 and standard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the size of the sample interval of the
        scaling factor is sampled from

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=False)

def yscale_normal_channel(x: np.ndarray, magnitude: float = .1):
    """
    "
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    from a normal distribtution with mean 0 and standard deviation magnitude
    scaling factor per channel
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the size of the sample interval of the
        scaling factor is sampled from

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=True)

def yscale_uniform(x: np.ndarray, magnitude: float = .1):
    """
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    uniformly from [-.5*magnitude; .5*magnitude]
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, size of scaling factor sample interval

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=False, by_channel=False)

def yscale_uniform_channel(x: np.ndarray, magnitude: float = .1):
    """
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    uniformly from [-.5*magnitude; .5*magnitude]
    scaling factor per channel
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, size of scaling factor sample interval

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=False, by_channel=True)

### time noise

def _normalize_timesteps(timesteps):
    '''
    distorted timesteps in [0,..,seq_len]
    '''
#     timesteps = timesteps - np.expand_dims(timesteps[:,0], -1)
#     timesteps = timesteps.clone()
    # timesteps = timesteps.sub(timesteps[:,0].unsqueeze(-1))
    timesteps = timesteps-(np.expand_dims(timesteps[:,0], -1))

#     timesteps = timesteps/np.expand_dims(timesteps[:,-1], -1) * (timesteps.shape[1]-1)
    timesteps=timesteps/(np.expand_dims(timesteps[:,-1], -1)) * (timesteps.shape[1]-1)

    return timesteps


def _distort_time(dim, magnitude=.1, smooth=False, **kwargs):
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
def _timenoise(x: np.ndarray, magnitude: float =.1, smooth: bool = False, **kwargs):
    '''
    helper function to add noise on the time axis, noise taken either from a normal distribution order
    a smoothed random curve
    basic approach: move data points along the time axis, e.g. from [0, 1, 2] to [0.4, 1.1, 1.8] along with 
    their corresponding y-values. then, use a CubicSpline to interpolate the y-values at the original
    points on the time axis, e.g. [0, 1, 2]
    '''

    if magnitude <= 0: return x
#     if len(x.shape)==1: x=x.unsqueeze(0) #no support for 1D tensors
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    n_channels, seq_len = x.shape[-2], x.shape[-1]
    # x_device = x.device ## make sure to put outpout on right device
    # x=x.cpu() ## only works on cpu
#    return f
#     plt.plot(x.T)
#     plt.plot(np.linspace(0,10), f(np.linspace(0,10)[:, None]).squeeze())
    # new_x = _distort_time((n_channels,seq_len), magnitude=magnitude, smooth=True, **kwargs).to(x.device)
    new_x = _distort_time((n_channels,seq_len), magnitude=magnitude, smooth=True, **kwargs)
    fs = [CubicSpline(np.arange(seq_len), x[...,i,:], axis=-1) for i in range(n_channels)]
#     new_y = f(new_x, )
#     print(fs(new_x).shape)
#     return new_x
    new_y = np.stack([fs[i](xi) for i,xi in enumerate(new_x)])
    if len(x.shape)==3: new_y = new_y.transpose(1,0,2)

    return new_y
    # return new_y.to(x_device, x.dtype)

def timewarp(x, magnitude=.1, order=4):
    '''
    distort time axis with random noise and interpolate values at original locations
    distortion noise from a smooth  random curve fitted on random poiunts sampled from
    a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    '''
    return _timenoise(x, magnitude, smooth=True, order=order)

def timenormal(x, magnitude=.1):
    '''
    distort time axis with random noise and interpolate values at original locations
    distortion noise from a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.ndarray of dimension (n_channels, seq_len)
    '''
    return _timenoise(x, magnitude, smooth=False)

#### zoom

def _randomize(p, mode='max'):
    '''
    how: 'max'|'min'
    '''
    # print(f'beta p {p}')
    p = np.random.beta(p,p)
    return np.maximum(p, 1-p) if mode=='max' else np.minimum(p, 1-p)

def _rand_steps(n, p, rand=False, window=False, mode='max'):
    if rand: p = _randomize(p, mode=mode)
    n_steps = int(p*n)
    if window:
        start = np.random.randint(0, n-n_steps+1)
        return np.arange(start, start+n_steps)
    else: return np.sort(np.random.choice(n, n_steps, replace=False))

def _zoom(x, magnitude=.2, rand=False, anchor=None, window=True, verbose=False):
    '''This is a slow batch tfm
    anchor: None|'left'|'right' 
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
    # print(window)
    if anchor=='right': window=np.arange(seq_len-len(window), seq_len) #right anchor
    if anchor=='left': window=np.arange(0, len(window)) #left anchor
    # print(anchor,window)
    # pv(window, verbose)
#     x2 = x[..., window]
    fs = [CubicSpline(np.arange(len(window)), x[...,i, window], axis=-1) for i in range(n_channels)]
    output = np.stack(
        [fs[i](np.linspace(0,len(window)-1, num=seq_len)) for i in range(n_channels)])
    if len(x.shape)==3: output = output.transpose(1,0,2)
#     output = x.new(f(np.linspace(0, len(window) - 1, num=seq_len)))

#     new_y = torch.stack([torch.tensor(fs[i](xi)) for i,xi in enumerate(new_x)])
#     if len(x.shape)==3: new_y = new_y.permute(1,0,2)

#     return new_y.to(x_device, x.dtype)
    return output
    # return output.to(x.device, x.dtype)

def zoom_in(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    zoom in augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: number of steps equal to (1-magnitude)*(seq_len)
    '''
    return partial(_zoom, rand=True, anchor=None, window=True)(x, magnitude=magnitude)


def zoom_right(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    zoom_out augementation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: number of steps equal to (1+magnitude)*(seq_len)
    '''
    return partial(_zoom, rand=True, anchor='right', window=True)(x, magnitude=magnitude)

def zoom_left(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    zoom_out augementation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: number of steps equal to (1+magnitude)*(seq_len)
    '''
    return partial(_zoom, rand=True, anchor='left', window=True)(x, magnitude=magnitude)

def rand_zoom(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    random zoom augementation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: number of steps equal to (1+magnitude)*(seq_len)
    '''
    p = np.random.rand()
    if p<=.333: anchor=None
    else:
        anchor='left' if  p<=.5 else 'right'
    return partial(_zoom, rand=True, anchor=anchor, window=True)(x, magnitude=magnitude)

def rand_timesteps(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    random time steps augementation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: number of steps equal to (1+magnitude)*(seq_len)
    '''
    # p = np.random.rand()
    # if p<=.333: anchor=None
    # else:
    #     anchor='left' if  p<=.5 else 'right'
    return partial(_zoom, rand=True, anchor=None, window=False)(x, magnitude=magnitude)

_zoomin = partial(_zoom, rand=True)
_zoomout = partial(_zoom, rand=True, zoomout=True)

def _randzoom(x, magnitude=.2):
    p = np.random.rand()
    return _zoomin(x, magnitude) if p<0.5 else _zoomout(x, magnitude)

### erasing, cropping augmentations

def _complement_steps(n, steps, verbose=False):
    # pv('complement', verbose)
    # pv(n, verbose)
    # pv(steps, verbose)
    if isinstance(n, int): n = np.arange(n)
    return np.sort(list(set(n)-set(steps)))

# Cell
def _center_steps(n, steps):
    start = n//2-len(steps)//2
    return np.arange(start, start+len(steps))

# Cell
def _create_mask_from_steps(x, steps, dim=False):
    '''create a 2D mask'''
    mask = np.zeros_like(x, dtype=bool)
    if len(steps)==0: return mask
    # print(f'mask shape {mask.shape}')
    # print(f'steps {steps}')
#     print(mask[steps,:])
#     print(mask[:, steps])
    if dim:
        mask[steps, :] = True
    else:
        mask[:, steps] = True
    return mask

# Cell
def _erase(x, magnitude=.2, rand=False, window=False, mean=False, complement=False, anchor=None, mask=False,
          dim=False, verbose=False, mode='max'):
    '''
    erasing parts of the timeseries
    anchor: None|'left'|'center'
    '''
    if magnitude==0: return x
    # pv(f'_erase input shape {x.shape}', verbose)
    assert len(x.shape)==2 or len(x.shape)==3, 'tensor needs to be 2D or 3D'
    is_batch = len(x.shape)==3
    # pv(x.shape, verbose)
    n_channels, seq_len = x.shape[-2], x.shape[-1]
    p = 1-magnitude if complement else magnitude

    n = n_channels if dim else seq_len
    steps = _rand_steps(n, p, rand=rand, window=window, mode=mode)
    if anchor=='left': steps = np.arange(len(steps))
    if anchor=='center': steps = _center_steps(n, steps)
    # print(steps)
    if complement: steps = _complement_steps(np.arange(n), steps, verbose=verbose)
    # print(steps)

    # pv(f'steps {steps}', verbose)
    # output = x.clone()
    output = x.copy()
    if not is_batch: output=np.expand_dims(output, 0)
    value = 0 if not mean else output.mean((0,2), keepdims=True)
    mask = np.random.rand(*output[0].shape)<magnitude if mask else _create_mask_from_steps(output[0], steps, dim=dim)
    # cutout=False
    # if cutout: mask = ~mask


    # pv(mask, verbose)
    # pv(value, verbose)

    if not mean: output[..., mask] = 0
    else:
        assert mask.shape[-2] == value.shape[-2]
        output[..., mask]=0
        output=output+np.expand_dims(mask.astype('int').astype(x.dtype), 0)*value
    # return output.squeeze_(0) if not is_batch else output
    return output
    # return np.expand_dims(output,0) if not is_batch else output

def timestep_zero(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    set the values of randomly selected time steps to zero
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=False, mean=False, complement=False, anchor=None, mask=False,
            dim=False)(x, magnitude=magnitude)

def timestep_mean(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    set the values of randomly selected time steps to the channel mean
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=False, mean=True, complement=False, anchor=None, mask=False,
            dim=False)(x, magnitude=magnitude)

def cutout(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    cutout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=True, mean=False, complement=False, anchor=None, mask=False,
            dim=False)(x, magnitude=magnitude)
    
def cutout_rand(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    cutout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=True, window=True, mean=False, complement=False, anchor=None, mask=False,
            dim=False, mode='min')(x, magnitude=magnitude)

def crop(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    cutout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=True, mean=False, complement=True, anchor=None, mask=False,
            dim=False)(x, magnitude=magnitude)

def left_crop(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    cutout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=True, mean=False, complement=True, anchor='left', mask=False,
            dim=False)(x, magnitude=magnitude)

def centercrop(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    cutout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=True, mean=False, complement=True, anchor='center', mask=False,
            dim=False)(x, magnitude=magnitude)
    
def maskout(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    maskout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=False, mean=False, complement=False, anchor='center', mask=True,
            dim=False)(x, magnitude=magnitude)

def dimout(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    dimout augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: time step erased with probability magnitude
    '''
    return partial(_erase, rand=False, window=False, mean=False, complement=False, anchor=None, mask=False,
            dim=True)(x, magnitude=magnitude)

_timestepzero = partial(_erase)
_timestepmean = partial(_erase, mean=True)
_cutout = partial(_erase, rand=True, window=True, complement=True)
_crop = partial(_erase, window=True, complement=True)
_randomcrop = partial(_erase, window=True, rand=True, complement=True)
_centercrop = partial(_erase, window=True, rand=True, anchor='center', complement=True)
_maskout = partial(_erase, mask=True)
_dimout = partial(_erase, dim=True)

def flip(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    flip augmentation
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: probability of flipping signs for each value
    '''
    flip_idxs = np.random.rand(*x.shape)<=magnitude
    return np.where(flip_idxs==0, -1, flip_idxs)*(-1)*x


def integer_noise(x: np.ndarray, magnitude:float = .1) -> np.ndarray:
    '''
    randomly add integers
    args:
        x: np.ndarray of dimension (n_channels, seq_len)
        magnitude: stdev of the normal distribution the noise is sampled from before rounding to the nearest int
    '''
    noise = integer_noise_from_normal((x.shape[-2], x.shape[-1]), magnitude=magnitude)
    # print(f'interger noise {noise}')
    output = x+noise
    return output

def quantize(x:np.ndarray, magnitude:float = .1, 
        n_levels:int = 10,
        how:str = 'quantile') -> np.ndarray:
    '''
    quantize x to levels
    n_levels: number of levels
    how: 'uniform': levels uniformly distributed over value range
        'quantile': levels correspond to quantiles of the values
    '''
    pass


def uniform_levels(x, n_levels=3):
    x_min, x_max = x.min(), x.max()
    # print(x_min, x_max)
    step = 1.*(x_max-x_min)/(n_levels)
    # print(step)
    bins = [i*step+0.5*step for i in range(n_levels)]
    return bins

def quantile_levels(x, n_levels=3):
    # x_min, x_max = x.min(), x.max()
    # print(x_min, x_max)
    step = 1/(n_levels)
    # print(step)
    bins = [np.quantile(x, i) for i in np.arange(0,10, 10/n_levels)]
    return bins
