from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .transforms import *




class BaseComposition(object):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        args:
            transforms: a list of transforms
            p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, (BaseTransform, BaseComposition)
            ), "Expected instances of type `BaseTransform` or `BaseComposition` for variable `transforms`"
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"

        self.transforms = transforms
        self.p = p


class Compose(BaseComposition):
    def __call__(self, x: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Applies the list of transforms in order to the time series
        args:
            x: the audio array to be augmented
            metadata: if set to be a list, metadata about the function execution

        returns: the augmented time series array 
        """
        for transform in self.transforms:
            # print(transform.f.func.__name__)
            x = transform(x, metadata)
        return x


class OneOf(BaseComposition):
    def __init__(self, transforms: List[BaseTransform], p: float = 1.0):
        """
        Select one transform from a list and apply it to the time series
        args:
            x: the audio array to be augmented
            metadata: if set to be a list, metadata about the function execution

        returns: the augmented time series array 
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(self, x: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Applies one of the transforms to the x (with probability p)
        args:
            x: the audio array to be augmented
            metadata: if set to be a list, metadata about the function execution

        returns: the augmented time series array 
        """
        if np.random.random() > self.p:
            return x

        transform = np.random.choice(self.transforms, p=self.transform_probs)
        return transform(x, metadata, force=True)

class RandAugment(BaseTransform):
    def __init__(self, magnitude=.1, p=1., N=2, tfms: List[str]=['noise']):
        super().__init__(p)
        if not isinstance(tfms, list): tfms = [tfms]
        assert len(tfms)>0, 'need at least one transform family'
        self.tfms = []
        for tfm in tfms:
            if tfm=='noise':
                self.tfms = self.tfms+all_noise_augs(magnitude=magnitude)
            if tfm=='ynoise':
                self.tfms = self.tfms+all_y_noise_augs(magnitude=magnitude)
            if tfm=='time_noise':
                self.tfms = self.tfms+all_time_noise_augs(magnitude=magnitude)
            if tfm=='scale':
                self.tfms = self.tfms+all_scale_augs(magnitude=magnitude)
            if tfm=='zoom':
                self.tfms = self.tfms+all_zoom_augs(magnitude=magnitude)
            if tfm=='erasing':
                self.tfms = self.tfms+all_erasing_augs(magnitude=magnitude)
            if tfm=='flip':
                self.tfms = self.tfms+[Flip(magnitude=magnitude)]
            if tfm=='integer_noise':
                self.tfms = self.tfms+[IntegerNoise(magnitude=magnitude)]
            if tfm=='all':
                self.tfms = self.tfms+all_augs(magnitude=magnitude)
            if tfm=='erasing_2':
                self.tfms = self.tfms+all_erasing_augs_v2(magnitude=magnitude)
            if tfm=='zoom_2':
                self.tfms = self.tfms+all_zoom_augs_v2(magnitude=magnitude)
            if tfm=='scale_2':
                self.tfms = self.tfms+all_scale_augs_v2(magnitude=magnitude)
            if tfm=='all_2':
                self.tfms = self.tfms+all_augs_v2(magnitude=magnitude)
            if tfm=='no_warp':
                self.tfms = self.tfms+all_no_warp_augs(magnitude=magnitude)
        self.N = N

    def apply_transform(self, x:np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        # print(f'apply_transform {self.tfms}')
        _tfms = np.random.choice(self.tfms, self.N)
        # print('apply tfms')
        # for tfm in _tfms: print(tfm.f.func.__name__)
        return Compose(_tfms)(x)
        # return x
