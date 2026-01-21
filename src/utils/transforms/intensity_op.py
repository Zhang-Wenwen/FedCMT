import numpy as np
from typing import Hashable, Mapping, List
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import Transform, MapTransform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor


class ClipIntensityRangePerChannel(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: List[float],
        a_max: List[float],
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        for channel_idx in range(img.shape[0]):
            img[channel_idx, ::] = clip(img[channel_idx, ::], self.a_min[channel_idx], self.a_max[channel_idx])
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret


class ClipIntensityRangePerChanneld(MapTransform):
    backend = ClipIntensityRangePerChannel.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: List[float],
        a_max: List[float],
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ClipIntensityRangePerChannel(a_min, a_max, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d
    

