"""
    Adapt nn-UNet's experiment planner to fixed k-divisible shape.

    I will not use nn-UNet's searched network architecture.

    Modifications:
    1. fixed architecture
    2. different path protocol
    3. disabled 2D configuration
"""


from __future__ import annotations
import os
import shutil
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Union, Dict, Any
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import _maybe_copy_splits_file, ExperimentPlanner
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from .path import get_paths


class CustomizedExperimentPlanner(ExperimentPlanner):
    dataset_json: Dict[str, Dict[str, Any]]

    def __init__(
        self, 
        dataset_name_or_id: int, 
        gpu_memory_target_in_gb: float = 8, 
        preprocessor_name: str = 'CustomizedPreprocessor', 
        plans_name: str = 'nnUNetPlansKDivisible', 
        overwrite_target_spacing: List[float] | Tuple[float, ...] = None, 
        suppress_transpose: bool = False,
        path_conf: Dict[str, str] | str = "./config/preprocessing/default.json"
    ):
        self.suppress_transpose = suppress_transpose
        
        # my customized way of PATH setting
        self.raw_dataset_folder, preprocessed_folder, _ = get_paths(path_conf, dataset_name_or_id)
        self.preprocessed_folder = preprocessed_folder
        self.dataset_name = os.path.basename(self.raw_dataset_folder)

        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.raw_dataset_folder, self.dataset_json)

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, 'dataset_fingerprint.json')):
            raise RuntimeError('Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint')

        self.dataset_fingerprint = load_json(join(preprocessed_folder, 'dataset_fingerprint.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce nnU-Net v1's configurations as
        # much as possible
        self.UNet_reference_val_3d = 560000000  # 455600128  550000000
        self.UNet_reference_val_2d = 85000000  # 83252480
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320
        self.max_dataset_covered = 0.05 # we limit the batch size so that no more than 5% of the dataset can be seen
        # in a single forward/backward pass

        self.UNet_vram_target_GB = gpu_memory_target_in_gb

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_identifier = plans_name
        self.overwrite_target_spacing = overwrite_target_spacing
        assert overwrite_target_spacing is None or len(overwrite_target_spacing), 'if overwrite_target_spacing is ' \
                                                                                  'used then three floats must be ' \
                                                                                  'given (as list or tuple)'
        assert overwrite_target_spacing is None or all([isinstance(i, float) for i in overwrite_target_spacing]), \
            'if overwrite_target_spacing is used then three floats must be given (as list or tuple)'

        self.plans = None

        if isfile(join(self.raw_dataset_folder, 'splits_final.json')):
            _maybe_copy_splits_file(join(self.raw_dataset_folder, 'splits_final.json'),
                                    join(preprocessed_folder, 'splits_final.json'))

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

        # Modification: fixed model architecture
        # I tried to modify 
        pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        shape_must_be_divisible_by = np.array(self.shape_must_be_divisible_by)
        patch_size = (np.round(initial_patch_size / shape_must_be_divisible_by) * shape_must_be_divisible_by + 0.1).astype(int)
        num_stages = len(pool_op_kernel_sizes)

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        # we enforce a batch size of at least two, reference values may have been computed for different batch sizes.
        # Correct for that in the while loop if statement
        while (estimate / ref_bs * 2) > reference:
            axis_to_be_reduced = np.argsort([(i / j if i - shape_must_be_divisible_by[dim_idx] > 0 else 0) for dim_idx, (i, j) in enumerate(zip(patch_size, median_shape[:len(spacing)]))])[-1]

            patch_size = list(patch_size)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            num_stages = len(pool_op_kernel_sizes)
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan
    
    def plan_experiment(self, shape_must_be_divisible_by: np.ndarray):
        """
            I am not intending to set default for `shape_must_be_divisible_by` so that I can be informed of this param.
        """
        self.shape_must_be_divisible_by = shape_must_be_divisible_by
        # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset, _tmp)
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                else:
                    lowres_spacing *= spacing_increase_factor
                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                # print(lowres_spacing)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  tuple([round(i) for i in plan_3d_fullres['spacing'] /
                                                                         lowres_spacing * new_median_shape_transposed]),
                                                                  self.generate_data_identifier('3d_lowres'),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json['numTraining']), _tmp)
                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
            if np.prod(new_median_shape_transposed, dtype=np.float64) / median_num_voxels < 2:
                print(f'Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. '
                      f'3d_fullres: {new_median_shape_transposed}, '
                      f'3d_lowres: {[round(i) for i in plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed]}')
                plan_3d_lowres = None
            if plan_3d_lowres is not None:
                plan_3d_lowres['batch_dice'] = False
                plan_3d_fullres['batch_dice'] = True
            else:
                plan_3d_fullres['batch_dice'] = False
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # # 2D configuration
        # plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
        #                                            new_median_shape_transposed[1:],
        #                                            self.generate_data_identifier('2d'), approximate_n_voxels_dataset,
        #                                            _tmp)
        # plan_2d['batch_dice'] = True

        # print('2D U-Net configuration:')
        # print(plan_2d)
        # print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(self.preprocessed_folder, 'dataset.json'))

        # json is ###. I hate it... "Object of type int64 is not JSON serializable"
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': {}},
            # 'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        if plan_3d_lowres is not None:
            plans['configurations']['3d_lowres'] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
            print('3D lowres U-Net configuration:')
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans['configurations']['3d_cascade_fullres'] = {
                    'inherits_from': '3d_fullres',
                    'previous_stage': '3d_lowres'
                }

        self.plans = plans
        self.save_plans(plans)
        return plans
    
    def save_plans(self, plans):
        recursive_fix_for_json_export(plans)

        plans_file = join(self.preprocessed_folder, self.plans_identifier + '.json')

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(self.preprocessed_folder))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(self.preprocessed_folder, self.plans_identifier + '.json')}")