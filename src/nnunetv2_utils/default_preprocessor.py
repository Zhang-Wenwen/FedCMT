"""
    Integrates pre-processing and inverse pre-processing in the same object.

    Also change the PATH convention.
"""


from __future__ import annotations
import shutil
import multiprocessing
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from time import sleep
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import isdir, join, isfile, load_json, maybe_mkdir_p
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from .path import get_paths


class nnUNetComponentUtils:
    """
        Manage a set of different datasets; they will use different configurations used in other components.

        Split it from preprocessor.
    """
    plans_managers: Dict[int, PlansManager] = {}
    configuration_managers: Dict[int, ConfigurationManager] = {}
    dataset_jsons: Dict[int, Any] = {}
    label_managers: Dict[int, LabelManager] = {}
    raw_folders: Dict[int, str] = {}
    preprocessed_folders: Dict[int, str] = {}
    results_folders: Dict[int, str] = {}

    def __init__(self, dataset_ids: List[int], path_conf: str, plans_identifier: str = "nnUNetPlansKDivisible", configuration_name: str = "3d_fullres"):
        self.dataset_ids = dataset_ids
        self.plans_identifier = plans_identifier
        self.configuration_name = configuration_name
        preprocessor = CustomizedPreprocessor(path_conf, verbose=False)
        for dataset_id in dataset_ids:
            dataset_config = preprocessor._get_cached_configs(dataset_id, plans_identifier, configuration_name)
            self.plans_managers[dataset_id] = dataset_config["plans_manager"]
            self.configuration_managers[dataset_id] = dataset_config["configuration_manager"]
            self.dataset_jsons[dataset_id] = dataset_config["dataset_json"]
            self.label_managers[dataset_id] = dataset_config["label_manager"]
            self.raw_folders[dataset_id], self.preprocessed_folders[dataset_id], \
                self.results_folders[dataset_id] = get_paths(path_conf, dataset_id)

class CustomizedPreprocessor(DefaultPreprocessor):
    def __init__(self, path_conf:str, verbose: bool = True):
        super().__init__(verbose)
        self.path_conf = path_conf
        self.cached_configs = {}

    def run(self, dataset_id: int, configuration_name: str, plans_identifier: str,
            num_processes: int):
        raw_folder, preprocessed_folder, _ =  get_paths(self.path_conf, dataset_id)

        assert isdir(raw_folder), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(preprocessed_folder, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}', flush=True)
        if self.verbose:
            print(configuration_manager, flush=True)

        dataset_json_file = join(preprocessed_folder, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(preprocessed_folder, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(raw_folder, dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]

            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def _get_cached_configs(self, dataset_id: int, plans_identifier: str, configuration_name: str):
        if dataset_id not in self.cached_configs:
            _, preprocessed_folder, _ =  get_paths(self.path_conf, dataset_id)
            plans_file = join(preprocessed_folder, plans_identifier + '.json')
            assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                    "first." % plans_file
            plans = load_json(plans_file)
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration(configuration_name)
            dataset_json_file = join(preprocessed_folder, 'dataset.json')
            dataset_json = load_json(dataset_json_file)
            label_manager = LabelManager(dataset_json["labels"], regions_class_order=dataset_json.get("regions_class_order"))
            self.cached_config = {
                "plans_manager": plans_manager,
                "configuration_manager": configuration_manager,
                "dataset_json": dataset_json,
                "label_manager": label_manager,
            }
        return self.cached_config
    
    def preprocess_online(self, image_files: List[str], seg_file: str, dataset_id: int, plans_identifier: str = "nnUNetPlansKDivisible", configuration_name: str = "3d_fullres") -> Tuple[np.ndarray, np.ndarray, dict]:
        """
            Interface for online pre-processing. I think it should be easier to use than a MONAI composed transforms,
            as the information necessary for inverse transform are not saved in the transform instance. We may easily
            parallelize the pre-processing and post-processing.
            
            It can be ran in a loop, as I write a cache for it. Please note that do not change `plans_identifier` or
            `configuration_name` during the loop, as I have not write anything compatible with that.

            Params:
                images_files:       list of image file absolute paths
                seg_file:           segmentation file absolute paths
                dataset_id:         dataset ID for pre-processing configuration
                plans_identifier:   [Optional] plan JSON file name
                configuration_name: [Optional] nnUNet configuration name

            Returns:
                pre-processed image array, pre-processed segmentation array, properties required for an inverse transform
        """
        config = self._get_cached_configs(dataset_id, plans_identifier, configuration_name)
        return self.run_case(image_files, seg_file, config["plans_manager"], config["configuration_manager"], config["dataset_json"])
    
    def inverse_preprocess_online(self, prediction: torch.Tensor | np.ndarray, properties: dict, dataset_id: int, plans_identifier: str = "nnUNetPlansKDivisible", configuration_name: str = "3d_fullres"):
        """
            Interface for online inverse pre-processing. Logits ranging in [0, 1] and summing to 1 channel-wise are 
            transformed back to original label shape.

            It can be ran in a loop, as I write a cache for it. Please note that do not change `plans_identifier` or
            `configuration_name` during the loop, as I have not write anything compatible with that.

            Params:
                prediction:         predicted logits; no need to do `sigmoid` or `softmax` or anything alike before feeding into this function
                properties:         pre-processing properties
                dataset_id:         dataset ID for pre-processing configuration
                plans_identifier:   [Optional] plan JSON file name
                configuration_name: [Optional] nnUNet configuration name

            Returns:
                reverted prediction
        """
        config = self._get_cached_configs(dataset_id, plans_identifier, configuration_name)
        return convert_predicted_logits_to_segmentation_with_correct_shape(prediction, config["plans_manager"], config["configuration_manager"], config["label_manager"], properties, return_probabilities=False)

