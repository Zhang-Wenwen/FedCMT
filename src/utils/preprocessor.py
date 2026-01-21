from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, Any, List, Callable
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (Transform, Compose, LoadImaged, EnsureChannelFirstd, Orientationd, CropForegroundd, 
                              NormalizeIntensityd, Spacingd, Invertd, AsDiscreted, SaveImaged)
from monai.utils import ImageMetaKey
from .path import get_paths
from .utils import spawn_processes
from .transforms.intensity_op import ClipIntensityRangePerChanneld
from .transforms.label_op import PermuteLabelsd


def select_nonzero(img): return img != 0


def save_file_name_formatter(metadict: dict, saver: Transform) -> dict:
    subject = (
        metadict.get(ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0)).replace("_0000.", ".")
        if metadict
        else getattr(saver, "_data_index", 0)
    )
    print("subject", subject)
    patch_index = metadict.get(ImageMetaKey.PATCH_INDEX, None) if metadict else None
    return {"subject": f"{subject}", "idx": patch_index}


def parse_preproc_transforms(cmd:Dict[str, Any], light_weighted: bool = False) -> Callable:
    transforms = [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        Orientationd(keys=["image","label"], axcodes="RAS", allow_missing_keys=True),
        CropForegroundd(keys=["image", "label"], source_key="image", select_fn=select_nonzero)
    ]

    # intensity operation
    if cmd["intensity clipping"] is not None:
        transforms.append(ClipIntensityRangePerChanneld(keys=["image"], a_min=cmd["intensity clipping"]["percentile_00_5"], a_max=cmd["intensity clipping"]["percentile_99_5"], allow_missing_keys=True))
    transforms.append(NormalizeIntensityd(keys="image", subtrahend=cmd["z-score norm"]["mean"], divisor=cmd["z-score norm"]["std"], channel_wise=True, nonzero=cmd["z-score norm"]["use_nonzero_for_norm"]))
    

    # resampling operation
    transforms.append(
        Spacingd(
            keys="image", 
            pixdim=cmd["resampling"]["image"]["pixdim"], 
            mode=cmd["resampling"]["image"]["mode"],
            padding_mode=cmd["resampling"]["image"]["padding_mode"],
            allow__keys=True
        )
    )

    if not light_weighted:
        # reorganize label maps
        transforms.append(PermuteLabelsd(keys=["label"], permuted_labels=cmd["labels"], allow_missing_keys=True))
        transforms.append(
            Spacingd(
                keys="label", 
                pixdim=cmd["resampling"]["label"]["pixdim"], 
                mode=cmd["resampling"]["label"]["mode"], 
                padding_mode=cmd["resampling"]["label"]["padding_mode"],
                allow_missing_keys=True
            )
        )
        transforms.append(
            AsDiscreted(
                keys=["label"],
                threshold=0.5,
                allow_missing_keys=True
            )
        )
        # organize the labels back
        transforms.append(
            PermuteLabelsd(keys=["label"], permuted_labels=cmd["labels"], allow_missing_keys=True, inverse=True)
        )
    else:
        transforms.append(
            Spacingd(
                keys="label", 
                pixdim=cmd["resampling"]["label"]["pixdim"], 
                mode="nearest", 
                padding_mode=cmd["resampling"]["label"]["padding_mode"],
                allow_missing_keys=True
            )
        )

    return Compose(transforms)


def wrap_list(x: List[int] | int): return x if isinstance(x, list) else [x]


def save_data_npy(data:Dict[str, Any], save_path:str):
    data = {k: data[k].get_array() if isinstance(data[k], MetaTensor) else data[k] for k in data}
    np.save(save_path, data, allow_pickle=True)


# avoid using closure...
class StatsCollector:
    def __init__(self, load_transforms: Callable, crop_fg_transform: Callable, num_samples: int):
        self.load_transforms = load_transforms
        self.crop_fg_transform = crop_fg_transform
        self.num_samples = num_samples
    
    def __call__(self, data_item:Dict[str, Any]):
        data_loaded = self.load_transforms(data_item)
        data_cropped = self.crop_fg_transform(data_loaded)
        fg_intensity_per_channel = {}
        for i in range(len(data_cropped["image"])):
            fg_pixels = data_cropped["image"][i:i+1][data_cropped["label"] > 0]
            fg_intensity_per_channel[i] = np.random.choice(fg_pixels, self.num_samples, replace=True).tolist() if len(fg_pixels) > 0 else []
        return {
            "cropped_shape": data_cropped["image"].shape[1:],
            "rel_size_after_cropping": np.prod(data_cropped["image"].shape[1:]) / np.prod(data_loaded["image"].shape[1:]),
            "pixdim": data_cropped["image"].pixdim,
            "fg_intensity_per_channel": fg_intensity_per_channel
        }


# avoid using closure...
class ProcessAndSaver:
    def __init__(self, preproc_transforms: Callable, save_folder:str, channel_idxs: List[int], replace_channel_idxs: bool):
        self.preproc_transforms = preproc_transforms
        self.save_folder = save_folder
        self.channel_idxs = channel_idxs
        self.replace_channel_idxs = replace_channel_idxs

    def __call__(self, img_dict):
        data_preproc = self.preproc_transforms(img_dict)
        basename: str = os.path.basename(img_dict["image"][0])
        if self.replace_channel_idxs:
            for channel_idx in self.channel_idxs: 
                replace_str = f"_{int(channel_idx):0>4d}."
                if replace_str in basename:
                    basename = basename.replace(replace_str, ".")
                    break
        save_path = os.path.join(self.save_folder, basename.split(".")[0] + ".npy")
        save_data_npy(data_preproc, save_path)


def _plan_softmax(x: List[List[int]]):
    merged = []
    for item in x:
        if len(item) > 1: return False
        merged.extend(item)
    merged_unique = np.unique(merged).tolist()
    return len(merged) == len(merged_unique)


def compute_new_shape(old_shape: np.ndarray, old_spacing: np.ndarray, new_spacing: np.ndarray):
    """
        Reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/resampling/default_resampling.py#L25
    """
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


class PreprocessManager:
    def __init__(
        self,
        dataset_id: int,
        num_processes: int,
        path_conf: Dict[str, str],
        verbose: bool,
        light_weighted: bool = False
    ) -> None:
        """
            Pre-processing manager class

            Params:
            ----------
                dataset_id:     the corresponding dataset id as stored in `nnUNet_raw` folder
                num_processes:  number of processes used for pre-processing
                path_conf:      path to a JSON configuration file containing information like `nnUNet_raw` folder; see README.md for more details
                verbose:        whether to disable progress bars
                light_weighted: if set True, will use nearest interpolation for labels so that less memory is required
        """
        self._dataset_folder, self._preprocess_folder, self._prediction_folder = get_paths(path_conf, dataset_id)
        self._num_processes = num_processes
        self._verbose = verbose
        self._light_weighted=  light_weighted
        self._aniso_threshold = 3
        
        with open(os.path.join(self._dataset_folder, "dataset.json"), "r") as fp:
            self._dataset_json: Dict[str, Any] = json.load(fp)
        if os.path.exists(os.path.join(self._preprocess_folder, "dataset.json")):
            # If a preprocessed `dataset.json` file already exists, use it instead
            with open(os.path.join(self._preprocess_folder, "dataset.json"), "r") as fp:
                self._dataset_json: Dict[str, Any] = json.load(fp)

        dataset_folder: str = os.path.basename(self._dataset_folder)
        self._dataset_name = dataset_folder.replace(dataset_folder.split("_")[0] + "_", "")

        self._preproc_transform = None
        self._postproc_transform = None
        self._replace_channel_idx = False

    def _collect_image_label_dicts_with_list(self, filenames:List[str], subfolders:List[str]):
        samples:List[Dict[str, Any]] = []
        file_ending = self._dataset_json["file_ending"]
        for filename in sorted(filenames):
            img_file_names = [
                os.path.join(self._dataset_folder, subfolders[0], filename.replace(file_ending, f"_{int(channel_idx):0>4d}{file_ending}")) \
                for channel_idx in self._dataset_json["channel_names"]
            ]
            if not os.path.exists(img_file_names[0]) and len(self._dataset_json["channel_names"]) == 1:
                img_file_names = [os.path.join(self._dataset_folder, subfolders[0], filename)]
            for img_file_name in img_file_names: assert os.path.exists(img_file_name), f"Missing file {img_file_name}"
            sample = {"image": img_file_names}
            if len(subfolders) > 1: sample["label"] = os.path.join(self._dataset_folder, subfolders[1], filename)
            samples.append(sample)
        return samples
    
    def _collect_preprocessed_paths_with_list(self, filenames: List[str], subfolder: str):
        return [os.path.join(self._preprocess_folder, subfolder, filename) for filename in filenames]
    
    def collect_preprocesssed_paths(self, subfolder: str = "imagesTr", split_val: bool = False):
        assert isinstance(subfolder, str), f"subfolders should be a string, got {subfolder}"
        filenames = [filename for filename in os.listdir(os.path.join(self._preprocess_folder, subfolder))]
        if not split_val:
            return self._collect_preprocessed_paths_with_list(filenames, subfolder)
        preset_val = self._dataset_json.get("preset_validation", [])
        preset_val = [filename + ".npy" for filename in preset_val]
        train_set = [filename for filename in filenames if filename not in preset_val]
        val_set = [filename for filename in filenames if filename in preset_val]

        # different from nn-UNetv2: in case that the training sample is extremenly few, we may use the full training set for validation?
        if len(train_set) == 0: train_set = val_set

        return self._collect_preprocessed_paths_with_list(train_set, subfolder), self._collect_preprocessed_paths_with_list(val_set, subfolder)
    
    def collect_image_label_dicts(self, subfolders:List[str]=["imagesTr", "labelsTr"], split_val:bool=False):
        """
            Label subfolders can be missing
            If `split_val` is assigned True, then there will be an additional return value indicating the validation split
        """
        assert isinstance(subfolders, list), f"subfolders should be a list, got {subfolders}"
        filenames = os.listdir(os.path.join(self._dataset_folder, subfolders[0]))
        self._replace_channel_idx = False
        if len(self.dataset_json["channel_names"]) > 0 and all([any([f"_{int(channel_id):0>4d}." in filename for channel_id in self.dataset_json['channel_names'].keys()]) for filename in filenames]):
            filenames = [filename.replace("_" + filename.split(".")[0].split("_")[-1] + ".", ".") for filename in os.listdir(os.path.join(self._dataset_folder, subfolders[0]))]
            self._replace_channel_idx = True
        filenames = list(set(filenames))
        if not split_val:
            return self._collect_image_label_dicts_with_list(filenames, subfolders)
        preset_val = self._dataset_json.get("preset_validation", [])
        preset_val = [filename + self._dataset_json["file_ending"] for filename in preset_val]
        train_set = [filename for filename in filenames if filename not in preset_val]
        val_set = [filename for filename in filenames if filename in preset_val]
        return self._collect_image_label_dicts_with_list(train_set, subfolders), self._collect_image_label_dicts_with_list(val_set, subfolders)

    def _collect_dataset_stats(self):
        img_lbl_dicts = self.collect_image_label_dicts()
        assert len(img_lbl_dicts) == self._dataset_json["numTraining"], f"Dataset integrity check failed, found {len(img_lbl_dicts)} samples instead of {self._dataset_json['numTraining']}"

        num_samples = 10000
        load_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
            ]
        )
        crop_fg_transform = CropForegroundd(keys=["image", "label"], source_key="image", select_fn=select_nonzero) # lambda expression is not allowed here...

        ds_stats = spawn_processes(StatsCollector(load_transforms, crop_fg_transform, num_samples), img_lbl_dicts, num_processes=self._num_processes, verbose=self._verbose, desc="StatsCollector")
        fg_intensity_per_channel, spaings_dist, cropped_shape_dist, rel_size = {}, [], [], []
        for ds_stat in ds_stats:
            spaings_dist.append(ds_stat["pixdim"])
            cropped_shape_dist.append(ds_stat["cropped_shape"])
            rel_size.append(ds_stat["rel_size_after_cropping"])
            for k in ds_stat["fg_intensity_per_channel"]:
                if k not in fg_intensity_per_channel: fg_intensity_per_channel[k] = []
                fg_intensity_per_channel[k].extend(ds_stat["fg_intensity_per_channel"][k])
        
        percentile_00_5, percentile_99_5, glob_fg_mean, glob_fg_std = {}, {}, {}, {}
        for k in fg_intensity_per_channel:
            percentile_00_5[k], percentile_99_5[k] = np.percentile(fg_intensity_per_channel[k], (0.5, 99.5))
            glob_fg_mean[k], glob_fg_std[k] = np.mean(fg_intensity_per_channel[k]), np.std(fg_intensity_per_channel[k])
        
        return np.stack(spaings_dist), np.stack(cropped_shape_dist), np.median(rel_size, 0), percentile_00_5, percentile_99_5, glob_fg_mean, glob_fg_std
    
    def _plan_spacing(self, spacings_dist:np.ndarray, cropped_shape_dist:np.ndarray) -> List[float]:
        """
            Spacing planning strategy, the same as nn-UNetv2
            See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py#L155
        """
        target:np.ndarray = np.percentile(spacings_dist, 50, 0)
        target_size = np.percentile(cropped_shape_dist, 50, 0)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self._aniso_threshold * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self._aniso_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = spacings_dist[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target.tolist()
    
    def _plan_normalization(self, percentile_00_5: Dict[str, float], percentile_99_5: Dict[str, float], g_fg_mean: Dict[str, float], g_fg_std: Dict[str, float], med_rel_size: float):
        modalities = self._dataset_json["channel_names"]

        #TODO: Guess there should not be mixtures of CT and other modalities?...
        is_ct = any([modalities[m] in ["ct"] for m in modalities])
        use_fg_for_norm = med_rel_size < 0.75
        mod_keys = sorted(list(percentile_00_5.keys()))
        
        cmd = {
            "intensity clipping": {
                "percentile_00_5": [percentile_00_5[k] for k in mod_keys],
                "percentile_99_5": [percentile_99_5[k] for k in mod_keys]
            } if is_ct else None,
            "z-score norm": {
                "mean": [g_fg_mean[k] for k in mod_keys] if is_ct else None,
                "std": [g_fg_std[k] for k in mod_keys] if is_ct else None,
                "use_nonzero_for_norm": bool(use_fg_for_norm)
            },
        }
        return cmd

    def _plan_patch_size_roughly(self, spacing: List[float], spacing_dist:np.ndarray, cropped_shape_dist: np.ndarray):
        max_pixels = 192 * 192 * 192 / 2
        shape_must_be_divisible_by = np.array([32, 32, 32])

        initial_patch_size = 1. / np.array(spacing)
        new_shapes = [compute_new_shape(j, i, spacing) for i, j in zip(spacing_dist, cropped_shape_dist)]
        median_shape = np.percentile(new_shapes, 50, 0)
        initial_patch_size = [round(i) for i in initial_patch_size * (256 ** 3 / np.prod(initial_patch_size)) ** (1 / 3)]
        patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])
        patch_size = (np.round(patch_size.astype(float) / shape_must_be_divisible_by) * shape_must_be_divisible_by).astype(int)
        while np.prod(patch_size) > max_pixels:
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
        
        return patch_size, int(max_pixels / np.prod(patch_size)) * 2

    def plan_preprocess_cmd(self):
        spacings_dist, cropped_shape_dist, median_rel_size, percentile_00_5, percentile_99_5, glob_fg_mean, glob_fg_std = self._collect_dataset_stats()

        use_softmax = _plan_softmax([wrap_list(self._dataset_json["labels"][label_name]) for label_name in sorted(self._dataset_json["labels"])])
        self._dataset_json["softmax"] = use_softmax

        spacing = self._plan_spacing(spacings_dist, cropped_shape_dist)
        cmd = {
            "softmax": use_softmax,
            "labels": [wrap_list(self._dataset_json["labels"][label_name]) for label_name in sorted(self._dataset_json["labels"]) if not (not use_softmax and label_name in ["background"])],
            "labels_name": [label_name for label_name in sorted(self._dataset_json["labels"]) if not (not use_softmax and label_name in ["background"])],
            "resampling": {
                "image": {
                    "mode": 3,
                    "padding_mode": "nearest",
                    "pixdim": spacing
                },
                "label": {
                    "mode": 1,
                    "padding_mode": "nearest",
                    "pixdim": spacing
                }
            }
        }
        self._dataset_json["preprocessed_label_order"] = cmd["labels_name"]

        norm_cmd = self._plan_normalization(percentile_00_5, percentile_99_5, glob_fg_mean, glob_fg_std, median_rel_size)
        for k in norm_cmd: cmd[k] = norm_cmd[k]

        cmd["patch_size"], cmd["num_patches"] = self._plan_patch_size_roughly(spacing, spacings_dist, cropped_shape_dist)
        cmd["patch_size"] = cmd["patch_size"].tolist()
        
        with open(os.path.join(self._preprocess_folder, "plan.json"), "w") as fp:
            json.dump(cmd, fp, indent=4)

        return cmd
    
    def preprocess_offline(self, cmd: Dict[str, Any] | None = None, subfolders: List[str] = ["imagesTr", "labelsTr"]):
        if cmd is None:
            with open(os.path.join(self._preprocess_folder, "plan.json"), "r") as fp:
                cmd = json.load(fp)
        
        img_lbl_dicts = self.collect_image_label_dicts(subfolders=subfolders)
        preproc_transforms = parse_preproc_transforms(cmd, self._light_weighted)

        save_folder = os.path.join(self._preprocess_folder, subfolders[0])
        os.makedirs(save_folder, exist_ok=True)
        
        spawn_processes(ProcessAndSaver(preproc_transforms, save_folder, list(self.dataset_json["channel_names"].keys()), self._replace_channel_idx), img_lbl_dicts, num_processes=self._num_processes, verbose=self._verbose, desc="ProcessAndSaver")

    def plan_and_preprocess(self, process_all: bool = False):
        cmd = self.plan_preprocess_cmd()
        self.preprocess_offline(cmd)
        if process_all:
            self.preprocess_offline(cmd, subfolders = ["imagesTs", "labelsTs"])
        with open(os.path.join(self._preprocess_folder, "dataset.json"), "w") as fp:
            json.dump(self._dataset_json, fp, indent=4)
    
    @property
    def preproc_cmd(self) -> Dict[str, Any] | None:
        plan_path = os.path.join(self._preprocess_folder, "plan.json")
        fp = open(os.path.join(self._preprocess_folder, "plan.json"), "r") if os.path.exists(plan_path) else None
        return json.load(fp) if fp is not None else None
    
    def _setup_transforms(self):
        if self._preproc_transform is None:
            cmd = self.preproc_cmd
            assert cmd is not None, "No plan JSON files found"
            self._preproc_transform = parse_preproc_transforms(cmd, self._light_weighted)
            self._postproc_transform = Compose(
                [
                    Invertd(
                        keys=["pred"],
                        transform=self._preproc_transform,
                        orig_keys=["image"],
                        nearest_interp=False
                    ),
                    #TODO:
                    AsDiscreted(keys=["pred"], threshold=None if cmd["softmax"] else 0.5, argmax=cmd["softmax"]),
                    PermuteLabelsd(keys=["pred"], permuted_labels=cmd["labels"], allow_missing_keys=True, inverse=True),
                ]
            )
    
    @property
    def preproc_transform(self) -> Compose: 
        if self._preproc_transform is None: self._setup_transforms()
        return self._preproc_transform

    @property
    def postproc_transform(self) -> Compose: 
        if self._postproc_transform is None: self._setup_transforms()
        return self._postproc_transform
    
    @property
    def dataset_name(self) -> str: return self._dataset_name

    @property
    def dataset_json(self) -> Dict[str, Any]: return self._dataset_json

    def preprocess(self, img_dict: Dict[str, List[str] | str]):
        return self.preproc_transform(img_dict)
    
    def load_label_from_dict(self, img_dict: Dict[str, List[str] | str]):
        return LoadImaged(keys=["label"])(img_dict)["label"]
    
    def postprocess(self, img_dict: Dict[str, Any], save: bool=False):
        """
            Prediction labels should be stored in key "pred", and keep the same number of channels as the preprocessed label
        """
        ret = self.postproc_transform(img_dict)
        if save:
            SaveImaged(
                keys=["pred"], 
                output_dir=self._prediction_folder, 
                output_postfix="", 
                output_dtype=np.int32, 
                output_ext=self._dataset_json["file_ending"], 
                separate_folder=False,
                output_name_formatter=save_file_name_formatter
            )(ret)
        return ret
