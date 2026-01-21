from __future__ import annotations
import numpy as np
import torch
from typing import List, Dict
from collections.abc import Callable, Hashable, Mapping
from monai.transforms.transform import MapTransform, Transform
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import TransformBackends
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.utils_pytorch_numpy_unification import in1d
from monai.transforms.transform import Transform
from monai.utils.misc import is_module_ver_at_least
from monai.utils.type_conversion import convert_to_tensor


def _check_invertible(permuted_labels: List[List[int]]):
    """
        Raises error if `permuted_labels` is not inversible
    """
    inverse_permute_dict: Dict[int, List[int]] = {}
    for channel_idx, target_labels in enumerate(permuted_labels):
        for target_label in target_labels:
            if target_label not in inverse_permute_dict: inverse_permute_dict[target_label] = []
            inverse_permute_dict[target_label].append(channel_idx)
    inverse_permute_list = [inverse_permute_dict[k] for k in inverse_permute_dict]
    for i in range(len(inverse_permute_list)):
        assert inverse_permute_list[i] not in inverse_permute_list[i + 1:], f"permuted_labels {permuted_labels} is not invertible"
    return inverse_permute_dict


class PermuteLabels(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, permuted_labels: List[List[int]], inverse: bool) -> None: 
        self.permuted_labels = permuted_labels
        self.inverse = inverse

    def __call__(
        self, img: NdarrayOrTensor, permuted_labels: List[List[int]] | None = None
    ) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if permuted_labels is None:
            permuted_labels = self.permuted_labels
        
        zeros = np.zeros if isinstance(img, np.ndarray) else torch.zeros
        if not self.inverse:
            data = zeros((len(permuted_labels), *img.shape[1:]), dtype=img.dtype)
            for channel_idx, target_labels in enumerate(permuted_labels):
                if img.shape[0] > 1:
                    heatmap = img[[*target_labels]]
                else:
                    where: Callable = np.where if isinstance(img, np.ndarray) else torch.where  # type: ignore
                    if isinstance(img, np.ndarray) or is_module_ver_at_least(torch, (1, 8, 0)):
                        heatmap = where(in1d(img, target_labels), True, False).reshape(img.shape)
                    # pre pytorch 1.8.0, need to use 1/0 instead of True/False
                    else:
                        heatmap = where(
                            in1d(img, target_labels), torch.tensor(1, device=img.device), torch.tensor(0, device=img.device)
                        ).reshape(img.shape)
                data[channel_idx:channel_idx + 1][heatmap] = 1
        else:
            ones = np.ones if isinstance(img, np.ndarray) else torch.ones
            data = zeros((1, *img.shape[1:]), dtype=img.dtype)

            if img.shape[0] == 1:
                data[:] = img
            
            else:
                # rearrange permuted_labels to {idx: channel} dict
                inverse_permute_dict = _check_invertible(permuted_labels)

                # rendering order is from the fewest elements (largest region) to most elements
                for idx in sorted(inverse_permute_dict, key=lambda k: len(inverse_permute_dict[k])):
                    mask = ones((1, *img.shape[1:]), dtype=bool)
                    for channel_idx in inverse_permute_dict[idx]: mask &= img[channel_idx: channel_idx + 1] > 0
                    data[mask] = idx
        
        return MetaTensor(data, meta=img.meta) if isinstance(img, MetaTensor) else data


class PermuteLabelsd(MapTransform):
    backend = PermuteLabels.backend

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: KeysCollection,
        permuted_labels: List[List[int]],
        inverse: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        """
            Note: this implementation is not good though, not implemented as dictionary form of a invertible transform
                Writing its inverse can be tricky, as we will probably be receiving floats for inverse inputs
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = PermuteLabels(permuted_labels=permuted_labels, inverse=inverse)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])

        return d