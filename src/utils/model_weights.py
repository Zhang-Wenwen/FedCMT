# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
import numpy as np
import torch
from loguru import logger
from utils.lora import LoRADynUNet
import torch.nn as nn


def load_weights(model: torch.nn.Module, weights: Dict[str, np.ndarray]) -> torch.nn.Module:
    local_var_dict = model.state_dict()
    aggregated_keys =  local_var_dict.keys() & weights.keys()

    for var_name in aggregated_keys:
        w = weights[var_name]
        local_var_dict[var_name] = torch.as_tensor(np.reshape(w, local_var_dict[var_name].shape))

    model.load_state_dict(local_var_dict)
    return model


def extract_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    local_state_dict = model.state_dict()
    local_model_dict = {}
    for var_name in local_state_dict:
        try:
            local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
        except Exception as e:
            raise ValueError(f"Convert weight from {var_name} failed with error: {str(e)}")
    return local_model_dict


def extract_encoder_weights(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    local_state_dict = model.state_dict()
    local_model_dict = {}
    encoder_keys = [key for key in local_state_dict if key.startswith('mod_channel') or key.startswith('downsamples.') or key.startswith('bottleneck.')]
    encoder_state_dict = {key: local_state_dict[key] for key in encoder_keys}
    for var_name in encoder_state_dict:
        local_model_dict[var_name] = encoder_state_dict[var_name].cpu().numpy()

    return local_model_dict

