from __future__ import annotations
import numpy as np
import torch
import warnings
from collections.abc import Sequence, Hashable, Collection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms import RandCropByPosNegLabel, RandCropByPosNegLabeld
from monai.transforms.utils import correct_crop_centers, check_non_lazy_pending_ops
from monai.transforms.utils_pytorch_numpy_unification import unravel_index, ravel, nonzero, any_np_pt
from monai.utils import ensure_tuple
from monai.utils.type_conversion import convert_data_type


def generate_pos_neg_label_crop_centers(
    spatial_size: Sequence[int] | int,
    num_samples_pos: int,
    num_samples_neg: int,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: np.random.RandomState | None = None,
    allow_smaller: bool = False,
) -> tuple[tuple]:
    """
        generates specific number of positive-centered and negative-centered samples
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        num_samples_pos, num_samples_neg = (0, num_samples_pos + num_samples_neg) if len(fg_indices) == 0 else (num_samples_pos + num_samples_neg, 0)
        warnings.warn(
            f"Num foregrounds {len(fg_indices)}, Num backgrounds {len(bg_indices)}, "
            f"unable to generate class balanced samples, setting `num_samples_pos` to {num_samples_pos}, and `num_samples_neg` to {num_samples_neg}."
        )

    for sample_idx in range(num_samples_pos + num_samples_neg):
        indices_to_use = fg_indices if sample_idx < num_samples_pos else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape).tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return ensure_tuple(centers)


def map_binary_to_indices(label: NdarrayOrTensor) -> tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
        1. Removes image threshold checking, as we are using normalized intensity
        2. Enables one-hot label input for region-based training
    """
    check_non_lazy_pending_ops(label, name="map_binary_to_indices")
    label_flat = ravel(any_np_pt(label, 0))  # in case label has multiple dimensions
    fg_indices = nonzero(label_flat)
    bg_indices = nonzero(~label_flat)

    # no need to save the indices in GPU, otherwise, still need to move to CPU at runtime when crop by indices
    fg_indices, *_ = convert_data_type(fg_indices, device=torch.device("cpu"))
    bg_indices, *_ = convert_data_type(bg_indices, device=torch.device("cpu"))
    return fg_indices, bg_indices


class CropByPosNegLabel(RandCropByPosNegLabel):
    """
        Non-probabilistic version (w.r.t. pos-neg choice) of RandCropByPosNegLabel
    """
    def __init__(
        self, 
        include_background: bool,
        spatial_size: Sequence[int] | int, 
        label: torch.Tensor | None = None, 
        pos: float = 1, 
        neg: float = 1, 
        num_samples: int = 1, 
        image: torch.Tensor | None = None, 
        image_threshold: float = 0, 
        fg_indices: np.ndarray | torch.Tensor | None = None, 
        bg_indices: np.ndarray | torch.Tensor | None = None, 
        allow_smaller: bool = False, 
        lazy: bool = False
    ) -> None:
        super().__init__(spatial_size, label, pos, neg, num_samples, image, image_threshold, fg_indices, bg_indices, allow_smaller, lazy)

        # non-probabilistic processing
        self.num_samples_pos = int(round(self.pos_ratio * self.num_samples))
        self.num_samples_neg = self.num_samples - self.num_samples_pos
        self.include_background = include_background
    
    def randomize(
        self,
        label: torch.Tensor | None = None,
        fg_indices: NdarrayOrTensor | None = None,
        bg_indices: NdarrayOrTensor | None = None,
        image: torch.Tensor | None = None,
    ) -> None:
        fg_indices_ = self.fg_indices if fg_indices is None else fg_indices
        bg_indices_ = self.bg_indices if bg_indices is None else bg_indices
        if fg_indices_ is None or bg_indices_ is None:
            if label is None:
                raise ValueError("label must be provided.")
            fg_indices_, bg_indices_ = map_binary_to_indices(label if self.include_background else label[1:])
        _shape = None
        if label is not None:
            _shape = label.peek_pending_shape() if isinstance(label, MetaTensor) else label.shape[1:]
        elif image is not None:
            _shape = image.peek_pending_shape() if isinstance(image, MetaTensor) else image.shape[1:]
        if _shape is None:
            raise ValueError("label or image must be provided to get the spatial shape.")
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples_pos,
            self.num_samples_neg,
            _shape,
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )
    

class CropByPosNegLabeld(RandCropByPosNegLabeld):
    """
        Non-probabilistic version of RandCropByPosNegLabeld
    """

    backend = CropByPosNegLabel.backend

    def __init__(
        self, 
        keys: Collection[Hashable] | Hashable, 
        include_background: bool,
        label_key: str, 
        spatial_size: Sequence[int] | int, 
        pos: float = 1, 
        neg: float = 1, 
        num_samples: int = 1, 
        image_key: str | None = None, 
        image_threshold: float = 0, 
        fg_indices_key: str | None = None, 
        bg_indices_key: str | None = None, 
        allow_smaller: bool = False, 
        allow_missing_keys: bool = False, 
        lazy: bool = False
    ) -> None:
        super().__init__(keys, label_key, spatial_size, pos, neg, num_samples, image_key, image_threshold, fg_indices_key, bg_indices_key, allow_smaller, allow_missing_keys, lazy)
        self.cropper = CropByPosNegLabel(
            include_background=include_background,
            spatial_size=spatial_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            lazy=lazy,
        )

    