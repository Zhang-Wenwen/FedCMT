from __future__ import annotations
import os
import numpy as np
from typing import Optional, Dict, Tuple, List
from torch import distributed as dist

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset

from nnunetv2_utils.data_loader import CustomizednnUNetDataLoader3D
from nnunetv2_utils.default_preprocessor import nnUNetComponentUtils

from loguru import logger
def get_allowed_n_proc_DA():
    return 6


class CustomizedDataManager:
    """
        Split from nnUNetTrainer.
    """
    def __init__(
        self,
        datasets_info: nnUNetComponentUtils, 
        do_unpack_dataset: bool = True,
    ):
        self.batch_size: Dict[int, int] = {}
        self.oversample_foreground_percent = 0.33

        # handle devices
        self.is_ddp = dist.is_available() and dist.is_initialized()

        # avoid copying codes...
        self.datasets_info = datasets_info

        for k in self.datasets_info.dataset_ids: self._set_batch_size_and_oversample(k)

        self.do_dummy_2d_data_aug: Dict[int, bool] = {}

        if do_unpack_dataset:
            for k in datasets_info.dataset_ids:
                print(f"unpacking dataset {k}...", flush=True)
                preprocessed_dataset_folder = os.path.join(datasets_info.preprocessed_folders[k], datasets_info.configuration_managers[k].data_identifier)
                unpack_dataset(preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
                print(f"unpacking dataset {k} done...", flush=True)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self, dataset_id: int):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.datasets_info.configuration_managers[dataset_id].patch_size
        dim = len(patch_size)
        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        # saves a log printing, otherwise the code is not good looking...
        self.do_dummy_2d_data_aug[dataset_id] = do_dummy_2d_data_aug
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def do_split(self, dataset_id: int):
        all_identifiers = get_case_identifiers(os.path.join(self.datasets_info.preprocessed_folders[dataset_id], f"{self.datasets_info.plans_identifier}_{self.datasets_info.configuration_name}"))
        val_keys = self.datasets_info.dataset_jsons[dataset_id]["preset_validation"]
        assert all([val_key in all_identifiers for val_key in val_keys]), f"some preset validation keys not found in the folder"
        tr_keys = [i for i in all_identifiers if i not in val_keys]
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self, dataset_id: int):
        tr_keys, val_keys = self.do_split(dataset_id)
        preprocessed_folder = os.path.join(self.datasets_info.preprocessed_folders[dataset_id], self.datasets_info.configuration_managers[dataset_id].data_identifier)
        dataset_tr = nnUNetDataset(preprocessed_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=None,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(preprocessed_folder, val_keys,
                                    folder_with_segs_from_previous_stage=None,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    @staticmethod
    def get_training_transforms(
        patch_size: np.ndarray | Tuple[int],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: List | Tuple | None,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Tuple[int, ...] | List[int] = None,
        regions: List[List[int] | Tuple[int, ...] | int] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False 
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:

            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: List | Tuple | None,
        is_cascaded: bool = False,
        foreground_labels: Tuple[int, ...] | List[int] = None,
        regions: List[List[int] | Tuple[int, ...] | int] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:

            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def _set_batch_size_and_oversample(self, dataset_id: int):
        if not self.is_ddp:

            self.batch_size[dataset_id] = self.datasets_info.configuration_managers[dataset_id].batch_size
        else:

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.datasets_info.configuration_managers[dataset_id].batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent, flush=True)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank], flush=True)

            self.batch_size[dataset_id] = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def get_dataloaders(self):
        """
            NOTE: I use `3d_fullres` only...

            Deep supervision is removed. Cascade option is disabled.
        """
        dataset_tr, dataset_val = {}, {}
        initial_patch_size = {}
        tr_transforms, val_transforms = {}, {}
        for k in self.datasets_info.dataset_ids:
            patch_size = self.datasets_info.configuration_managers[k].patch_size

            (
                rotation_for_DA,
                do_dummy_2d_data_aug,
                initial_patch_size[k],
                mirror_axes,
            ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size(k)

            # training pipeline
            tr_transforms[k] = self.get_training_transforms(
                patch_size, rotation_for_DA, None, mirror_axes, do_dummy_2d_data_aug,
                use_mask_for_norm=self.datasets_info.configuration_managers[k].use_mask_for_norm,
                is_cascaded=False, foreground_labels=self.datasets_info.label_managers[k].foreground_labels,
                regions=self.datasets_info.label_managers[k].foreground_regions if self.datasets_info.label_managers[k].has_regions else None,
                ignore_label=self.datasets_info.label_managers[k].ignore_label)

            # validation pipeline
            val_transforms[k] = self.get_validation_transforms(None,
                is_cascaded=False,
                foreground_labels=self.datasets_info.label_managers[k].foreground_labels,
                regions=self.datasets_info.label_managers[k].foreground_regions if
                self.datasets_info.label_managers[k].has_regions else None,
                ignore_label=self.datasets_info.label_managers[k].ignore_label)

            dataset_tr[k], dataset_val[k] = self.get_tr_and_val_datasets(k)

        dl_tr = CustomizednnUNetDataLoader3D(dataset_tr, self.batch_size,
            initial_patch_size,
            {k: self.datasets_info.configuration_managers[k].patch_size for k in self.datasets_info.dataset_ids},
            {k: self.datasets_info.label_managers[k] for k in self.datasets_info.dataset_ids},
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms)
        dl_val = CustomizednnUNetDataLoader3D(dataset_val, self.batch_size,
            {k: self.datasets_info.configuration_managers[k].patch_size for k in self.datasets_info.dataset_ids},
            {k: self.datasets_info.configuration_managers[k].patch_size for k in self.datasets_info.dataset_ids},
            {k: self.datasets_info.label_managers[k] for k in self.datasets_info.dataset_ids},
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=True, wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=True,
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

class DataManager:
    """
        I am not implementing `test` phase here!
    """
    def __init__(self, dataset_info: nnUNetComponentUtils) -> None:
        self.dataset_info = dataset_info
        self.nnunet_dm = CustomizedDataManager(dataset_info, do_unpack_dataset=False) # run once and disable it...
        self._data_loader = {}

    def setup(self, stage: Optional[str] = None):
        if stage not in ["training", "validation"]:
            raise NotImplementedError(f"DataManager not implemented for phase {stage} yet.")
        self._data_loader["training"], self._data_loader["validation"] = self.nnunet_dm.get_dataloaders()

    def get_data_loader(self, split:str): 
        return self._data_loader.get(split, None)

    def teardown(self): 
        self._data_loader = {}