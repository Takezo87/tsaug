from typing import Optional, List, Tuple, Any

import numpy as np
class BaseTransform(object):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(
        self,
        audio: np.ndarray,
        sample_rate: int = utils.DEFAULT_SAMPLE_RATE,
        metadata: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        @param audio: the audio array to be augmented
        @param sample_rate: the audio sample rate of the inputted audio
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest duration, sample rates, etc. will be
            appended to the inputted list. If set to None, no metadata will be appended
        @param force: if set to True, the transform will be applied. otherwise,
            application is determined by the probability set
        @returns: the augmented audio array and sample rate
        """
        assert isinstance(audio, np.ndarray), "Audio passed in must be a np.ndarray"
        assert type(force) == bool, "Expected type bool for variable `force`"

        if not force and np.random.random() > self.p:
            return audio, sample_rate

        return self.apply_transform(audio, sample_rate, metadata)

    def apply_transform(
        self,
        audio: np.ndarray,
        sample_rate: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()
