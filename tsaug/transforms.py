from typing import Optional, List, Tuple, Any, Dict
from functools import partial

import numpy as np
import functional as F

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
        assert isinstance(x, np.ndarray), "Timeseries passed in must be a np.ndarray"
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
                print(f'could not convert x of type {type(x) to np.ndarray}')




class YNoiseNormalAdd(BaseTransform):
    def __init__(self, magnitude=.1, p=1.):
        super().__init__(p)
        self.f = partial(F.ynoise_normal_add, magnitude=magnitude)

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        return self.f(x)

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

class 
            

