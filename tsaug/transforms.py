from typing import Optional, List, Tuple, Any, Dict, Union
from functools import partial

import numpy as np

import tsaug.functional as F


class BaseTransform:
    def __init__(self, p: float = 1.0):
        """
        p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(self, x: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None, force:bool = False)-> np.ndarray:
        """
        args:
            x: the timeseries array to be augmented
            metadata: if set to be a list, metadata about the function execution
                including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended
        returns:
            the augmented time series array
        """
        # assert isinstance(x, np.ndarray), "Timeseries passed in must be a np.ndarray"
        assert type(force) == bool, "Expected type bool for variable `force`"


        return self.apply_transform(x, metadata)
    

    def apply_transform(self, x: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()

class ToNumpy(BaseTransform):
    def __init__(self):
        super().__init__(p=1.)

    def apply_transform(self, x: Union[np.ndarray, Any],
        metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            try:
                return np.array(x)
            except:
                print(f'could not convert x of type {type(x)} to np.ndarray')

class ToLogit(BaseTransform):
    def __init__(self, format:str = 'decimal'):
        super().__init__(p=1.)
        assert format in ['decimal', 'probability']
        self.format = format

    def apply_transform(self, x: Union[np.ndarray, Any],
        metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return F.to_logit(x, format=self.format)
        

class YNoiseNormalAdd(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        '''
        args:
            magnitude:
        '''
        super().__init__(p)
        self.f = partial(F.ynoise_normal_add, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
    def __doc__(self):
        return self.f.__doc__()

class YNoiseNormalMul(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.ynoise_normal_mul, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class YNoiseNormalWarp(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.ynoise_normal_warp, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class TimeWarp(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.timewarp, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
    
class TimeNormal(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.timenormal, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

def all_noise_augs(magnitude=.1):
    return [YNoiseNormalWarp(magnitude=magnitude), YNoiseNormalAdd(magnitude=magnitude), 
            TimeWarp(magnitude=magnitude), TimeNormal(magnitude=magnitude)]

def all_y_noise_augs(magnitude=.1):
    return [YNoiseNormalWarp(magnitude=magnitude), YNoiseNormalAdd(magnitude=magnitude), 
            YNoiseNormalMul(magnitude=magnitude)]

def all_time_noise_augs(magnitude=.1):
    return [TimeWarp(magnitude=magnitude), TimeNormal(magnitude=magnitude)]

def all_augs(magnitude=.1):
    '''
    all augs except flip and integer_noise
    '''
    return all_noise_augs(magnitude=magnitude) + all_scale_augs(magnitude=magnitude) +all_zoom_augs(magnitude=magnitude) + all_erasing_augs(magnitude=magnitude)

ALL_NOISE = all_noise_augs(magnitude=.3)

####################
# scaling
####################

class YScaleNormal(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.yscale_normal, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class YScaleNormalChannel(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.yscale_normal_channel, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class YScaleUniform(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.yscale_uniform, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class YScaleUniformChannel(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.yscale_uniform_channel, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

def all_scale_augs(magnitude=.1):
    return [YScaleUniform(magnitude=magnitude), YScaleUniformChannel(magnitude=magnitude), 
            YScaleNormal(magnitude=magnitude), YScaleNormalChannel(magnitude=magnitude)]

ALL_SCALE = all_scale_augs(magnitude=.3)

####################
# zooming
####################
class ZoomIn(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.zoom_in, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class ZoomOut(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.zoom_out, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class RandZoom(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.rand_zoom, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
    
class RandTimesteps(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.rand_timesteps, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

def all_zoom_augs(magnitude=.1):
    return [ZoomIn(magnitude=magnitude), ZoomOut(magnitude=magnitude), 
            RandZoom(magnitude=magnitude), RandTimesteps(magnitude=magnitude)]

ALL_ZOOM = all_zoom_augs(magnitude=.3)

####################
# erasing
####################
class TimestepZero(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.timestep_zero, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class TimestepMean(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.timestep_mean, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class Cutout(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.cutout, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class Crop(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.crop, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class CenterCrop(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.centercrop, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
            
class Maskout(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.maskout, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

class Dimout(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.dimout, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

def all_erasing_augs(magnitude=.1):
    return [Maskout(magnitude=magnitude), Cutout(magnitude=magnitude), Crop(magnitude=magnitude), 
            CenterCrop(magnitude=magnitude), TimestepMean(magnitude=magnitude), Dimout(magnitude=magnitude),
            TimestepZero(magnitude=magnitude)]

ALL_ERASING = all_erasing_augs(magnitude=.3)

class Flip(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.flip, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
            

class IntegerNoise(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.integer_noise, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)
