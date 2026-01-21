"""
    As the task differs, I re-write the dataloader referencing nn-UNet
"""


from __future__ import annotations
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from typing import List, Tuple, Dict, Any, Callable
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class CustomizednnUNetDataLoader3D:
    """
        Will load data samples in a dataset-balanced way
    """
    def __init__(
        self, 
        data: Dict[int, nnUNetDataset], 
        batch_size: Dict[int, int], 
        patch_size: Dict[int, List[int]],
        final_patch_size: Dict[int, List[int]],
        label_manager: Dict[int, LabelManager],
        oversample_foreground_percent: float = 0.0,
        number_of_threads_in_multithreaded = 1, 
        seed_for_shuffle: Any | None = None, 
        return_incomplete: bool = False,
        shuffle: bool = True, 
        infinite: bool = True, 
        probabilistic_oversampling: bool = False,
        transforms: Dict[int, Callable | None] | None = None,
    ):
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0
        self.infinite = infinite
        self.shuffle = shuffle
        self.return_incomplete = return_incomplete
        self.seed_for_shuffle = seed_for_shuffle
        self.rs = np.random.RandomState(self.seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.last_reached = False
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.need_to_pad = {k: (np.array(patch_size[k]) - np.array(final_patch_size[k])).astype(int) for k in final_patch_size}
        self.annotated_classes_key = {k: tuple(label_manager[k].all_labels) for k in label_manager}
        self.has_ignore = {k: label_manager[k].has_ignore_label for k in label_manager}
        self.oversample_foreground_percent = oversample_foreground_percent
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

        self.indices = {k: list(data[k].keys()) for k in data}
        self.dataset_keys = sorted([k for k in data])

        self.data_shape, self.seg_shape = {}, {}
        for dataset_key in self.dataset_keys:
            self.data_shape[dataset_key], self.seg_shape[dataset_key] = self.determine_shapes(dataset_key)

    def reset(self):
        assert self.indices is not None

        self.current_dataset: int = 0
        self.current_position = self.thread_id * self.batch_size[self.dataset_keys[self.current_dataset]]

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        if not self.infinite and self.shuffle:
            for k in self.indices:
                self.rs.shuffle(self.indices[k])

        self.last_reached = False
    
    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            got_key = self.dataset_keys[np.random.randint(len(self.dataset_keys))]
            return np.random.choice(self.indices[got_key], self.batch_size[got_key], replace=True), got_key
        
        #TODO: if not infinite, the code can be a bit buggy, please do not use that

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []
        # not incremental position, and hence we need to check if it skips a dataset
        while self.current_dataset < len(self.dataset_keys) and self.current_position >= len(self.indices[self.dataset_keys[self.current_dataset]]):
            self.current_dataset += 1
            self.current_position -= len(self.indices[self.dataset_keys[self.current_dataset]])

        for b in range(self.batch_size[self.dataset_keys[self.current_dataset]]):
            if self.current_dataset < len(self.dataset_keys) and self.current_position < len(self.indices[self.dataset_keys[self.current_dataset]]):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
                if self.current_position == len(self.indices[self.dataset_keys[self.current_dataset]]):
                    # hmm, different datasets may not be able to collate batch
                    break
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size[self.dataset_keys[self.current_dataset]]
            return indices, self.dataset_keys[self.current_dataset]
        else:
            self.reset()
            raise StopIteration

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()
    
    def _oversample_last_XX_percent(self, sample_idx: int, dataset_key: str) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size[dataset_key] * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int, dataset_key: str) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent
    
    def determine_shapes(self, dataset_key: str):
        # load one case
        data, seg, properties = self._data[dataset_key].load_case(self.indices[dataset_key][0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size[dataset_key], num_color_channels, *self.patch_size[dataset_key])
        seg_shape = (self.batch_size[dataset_key], seg.shape[0], *self.patch_size[dataset_key])
        return data_shape, seg_shape
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, dataset_key: str, class_locations: dict | None,
                 overwrite_class: int | Tuple[int, ...] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad[dataset_key].copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[dataset_key][d]:
                need_to_pad[d] = self.patch_size[dataset_key][d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[dataset_key][i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore[dataset_key]:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore[dataset_key]:
                selected_class = self.annotated_classes_key[dataset_key]
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!', flush=True)
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key[dataset_key]}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key[dataset_key] if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes', flush=True)
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[dataset_key][i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[dataset_key][i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys, dataset_key = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape[dataset_key], dtype=np.float32)
        seg_all = np.zeros(self.seg_shape[dataset_key], dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j, dataset_key)

            data, seg, properties = self._data[dataset_key].load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, dataset_key, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None and self.transforms.get(dataset_key, None) is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size[dataset_key]):
                        tmp = self.transforms[dataset_key](**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'image': data_all, 'label': seg_all, 'keys': selected_keys, "dataset_id": dataset_key}

        return {'image': data_all, 'label': seg_all, 'keys': selected_keys, "dataset_id": dataset_key}