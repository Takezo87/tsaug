from typing import Optional, List, Tuple, Any

import numpy as np
from scipy.interpolate import CubicSpline

def _create_random_curve(n_channels, seq_len, magnitude, order):
    '''
    create a random curve for each channel in the interval[0, seq_len-1] on order random points
    '''
    x = np.linspace(0, seq_len-1, order)
    y = np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, len(x)))
    f = CubicSpline(x, y, axis=-1)
    return f

def noise_from_random_curve(dim: Tuple[int, int], magnitude: float = .1, order: int = 4) -> np.array:
    '''
    sample points from a gaussian with mean 1 and create a smooth cubic "random curve" from these points
    ts
    args:
        dim: dimension of timeseries array (n_channels, sequence lenght) 
        order: number of sample points to create the random curve from
        magnitude: standard deviation of the noise normal distribution

    returns:
        np.array of dimension dim
    '''
    n_channels, seq_len = dim
    f = _create_random_curve(n_channels, seq_len, magnitude, order)
    return f(np.arange(seq_len))

def noise_from_normal(dim: Tuple[int, int], magnitude: float = .1) -> np.array:
    '''
    sample random noise from a gaussian with mean=1.0 and std=magnitude
    args:
        dim: dimension of timeseries array (n_channels, sequence lenght) 
        magnitude: standard deviation of the noise normal distribution

    returns:
        np.array of dimension dim
    '''
    n_channels, seq_len = dim
    return np.random.normal(loc=1.0, scale=magnitude, size=(n_channels, seq_len))
