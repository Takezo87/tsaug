from typing import Optional, List, Tuple, Any
from functools import partial

import numpy as np
from scipy.interpolate import CubicSpline

from tsaug.utils import noise_from_normal, noise_from_random_curve

### y noise

def _ynoise(x: np.array, magnitude: float = .1, add: bool =True, smooth: bool =True, **kwargs):
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


def ynoise_normal_add(x: np.array, magnitude: float = .1):
    """
    add additive noise to timeseries x
    noise taken from normal distribution with mean 0. and stadard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of noise distribution

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _ynoise(x, magnitude=magnitude, add=True, smooth=False)

def ynoise_normal_mul(x: np.array, magnitude: float = .1):
    """
    add multiplicative noise to timeseries x
    noise taken from normal distribution with mean 1. and stadard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of noise distribution

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _ynoise(x, magnitude=magnitude, add=False, smooth=False)

def ynoise_normal_warp(x: np.array, magnitude: float = .1):
    """
    add additive noise to timeseries x
    noise taken from a smooth random curve fitted on random poiunts sampled from
    a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.array of dimension (n_channels, seq_len)
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
        scale = 1.+2*(np.random.randn(1 if not by_channel else x.shape[-2])-0.5)*magnitude
    else:
#         scale = 1 + torch.rand(1) * magnitude  # uniform [0:1], normal possible
        scale = 1.+2*(np.random.rand(1 if not by_channel else x.shape[-2])-0.5)*magnitude
#         if np.random.rand() < .5: scale = 1 / scale # scale down
#     output = x * scale.to(x.device)
    return x*scale.to(x.device) if not by_channel else x*scale[..., None].to(x.device)


def yscale_normal(x: np.array, magnitude: float = .1):
    """
    "
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    from a normal distribtution with mean 0 and standard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the size of the sample interval of the
        scaling factor is sampled from

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=False)

def yscale_normal_channel(x: np.array, magnitude: float = .1):
    """
    "
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    from a normal distribtution with mean 0 and standard deviation magnitude
    scaling factor per channel
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the size of the sample interval of the
        scaling factor is sampled from

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=True)

def yscale_uniform(x: np.array, magnitude: float = .1):
    """
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    uniformly from [-.5*magnitude; .5*magnitude]
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, size of scaling factor sample interval

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=False)

def yscale_uniform_channel(x: np.array, magnitude: float = .1):
    """
    scale all values of x by a random scaling factor from [1-a, 1+a], where a is chosen
    uniformly from [-.5*magnitude; .5*magnitude]
    scaling factor per channel
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, size of scaling factor sample interval

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    """
    return _yscale(x, magnitude=magnitude, normal=True, by_channel=False)

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
def _timenoise(x: np.array, magnitude: float =.1, smooth: bool = False, **kwargs):
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
    if len(x.shape)==3: new_y = new_y.permute(1,0,2)

    return new_y
    # return new_y.to(x_device, x.dtype)

# Cell
def timewarp(x, magnitude=.1, order=4):
    '''
    distort time axis with random noise and interpolate values at original locations
    distortion noise from a smooth  random curve fitted on random poiunts sampled from
    a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    '''
    return _timenoise(x, magnitude, smooth=True, order=order)

def timenormal(x, magnitude=.1):
    '''
    distort time axis with random noise and interpolate values at original locations
    distortion noise from a normal distribution with mean 0. and standard deviation magnitude
    args:
        x: np.array of dimension (n_channels, seq_len)
        magnitude: float, standard deviation of normal distribution the random points are sampled from

    returns:
        transformed np.array of dimension (n_channels, seq_len)
    '''
    return _timenoise(x, magnitude, smooth=False)
