from __future__ import annotations
import os
import json
from typing import Dict
from .enums import PATH_CONFIG_KEY


def _get_dataset_folder_name(raw_path:str, dataset_id:int):
    is_dataset_found, dataset_folder_name = False, None
    for folder_name in os.listdir(raw_path):
        if int(folder_name.split("_")[0][7:]) == dataset_id:
            if is_dataset_found:
                raise ValueError(f"There are more than one folders for dataset id {dataset_id}")
            is_dataset_found = True
            dataset_folder_name = folder_name
    return dataset_folder_name


def get_dataset_path(path_conf:Dict[str, str], dataset_id:int):
    return os.path.join(path_conf[PATH_CONFIG_KEY.RAW], _get_dataset_folder_name(path_conf[PATH_CONFIG_KEY.RAW], dataset_id))


def get_preprocessed_path(path_conf:Dict[str, str], dataset_id:int):
    ret = os.path.join(path_conf[PATH_CONFIG_KEY.PREPROCESSED], _get_dataset_folder_name(path_conf[PATH_CONFIG_KEY.RAW], dataset_id))
    os.makedirs(ret, exist_ok=True)
    return ret


def get_pred_path(path_conf:Dict[str, str], dataset_id:int):
    pred_folder = path_conf.get(PATH_CONFIG_KEY.PREDICT, None)
    if pred_folder is not None:
        ret = os.path.join(pred_folder, _get_dataset_folder_name(path_conf[PATH_CONFIG_KEY.RAW], dataset_id))
        os.makedirs(ret, exist_ok=True)
        return ret
    return None


def get_paths(path_conf: Dict[str, str] | str, dataset_id:int):
    if isinstance(path_conf, str): 
        with open(path_conf, "r") as fp:
            path_conf = json.load(fp)
    return get_dataset_path(path_conf, dataset_id), get_preprocessed_path(path_conf, dataset_id), get_pred_path(path_conf, dataset_id)