from __future__ import annotations

import os
import gc 
import torch
from copy import deepcopy
from typing import Dict, Any, Optional
from pathlib import Path, PurePath
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from losses import IcarlLoss, KnowledgeDistillationLoss
from utils.get_model import get_model
from nnunetv2_utils.data_loader import CustomizednnUNetDataLoader3D as DataLoader
from nnunetv2_utils.default_preprocessor import nnUNetComponentUtils
from loguru import logger
import copy

class LogSumExpPooling(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx: torch.autograd.Function, x: torch.Tensor, r: float = 5.0):
        exp_x = torch.exp((r * x).clamp(max=50)) 
        ctx.save_for_backward(exp_x, torch.ones((1,), device=x.device, dtype=float) * r)
        result = torch.log(torch.mean(exp_x, dim=list(range(2, len(x.shape)))).clamp(min=1e-6)) / r
        return result
    
    @staticmethod
    def backward(ctx: torch.autograd.Function, grad_outputs: torch.Tensor):
        exp_x, r, = ctx.saved_tensors
        broadcastable_shape = [*grad_outputs.shape, *([1] * (len(exp_x.shape) - 2))]
        local_grad: torch.Tensor = exp_x / torch.sum(exp_x, dim=list(range(2, len(exp_x.shape)))).reshape(*broadcastable_shape).clamp(min=1e-6)
        return grad_outputs.reshape(*broadcastable_shape) / r[0] * local_grad, None


@torch.no_grad()
def batch_label_map_to_cls_label(label_map: torch.Tensor, num_classes: int):
    label_map_one_hot: torch.Tensor = torch.nn.functional.one_hot(label_map.to(torch.int64).squeeze(1), num_classes=num_classes + 1)
    spatial_dims = list(range(1, len(label_map_one_hot.shape) - 1))
    label_one_hot = torch.sum(label_map_one_hot, dim=spatial_dims)
    label_one_hot = label_one_hot[:, 1:] 
    return torch.argmax(label_one_hot, dim=1)


class ClassificationLoss(torch.nn.Module):
    def __init__(self, has_region: bool, r: float = 5.0) -> None: #
        super().__init__()
        self.has_region = has_region
        self.lse = LogSumExpPooling.apply
        self.r = r
        self.loss = None if has_region else torch.nn.CrossEntropyLoss()

    
    def forward(self, predicted_map: torch.Tensor, label_map: torch.Tensor):
        predicted_logits = self.lse(predicted_map, self.r)
        if self.has_region:
            raise NotImplementedError("Not implemented region-based classification loss")
        
        label = batch_label_map_to_cls_label(label_map, num_classes=predicted_logits.shape[1])
        
        return self.loss(predicted_logits, label)


class Trainer:
    def __init__(self, task_config: Dict[str, Any | dict], dataset_info: nnUNetComponentUtils) -> None:
        self.init_lr = task_config["optimizer"]["lr"]
        self.weight_decay = task_config["optimizer"]["weight_decay"]
        self.model_config = task_config["model"]
        self.use_half_precision = task_config["training"].get("use_half_precision", False)
        self.dataset_info = dataset_info
        self.dataset_id = dataset_info.dataset_ids[0]

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_half_precision)

        self.loss_fun = self.get_loss()

        self.current_step = 0
        self.current_round = 0
        self.opt = None
        self.opt_state = None
        self.sch = None
        self.sch_state = None
        self.last_model_path = None
        self.icarl = 1
        self.licarl = IcarlLoss(reduction='mean', bkg=False)

        self.lkd = 1
        self.lkd_loss = KnowledgeDistillationLoss(alpha=1)



    def run(self, model: torch.nn.Module, data_loader: DataLoader, num_steps: int, logger, abort_signal: Optional[Any] = None):
        self.setup(model, logger, abort_signal)

        self.model.train()
        avg_loss = self.training_loop(data_loader, num_steps)
        self.current_round += 1

        self.cleanup()

        return avg_loss
        
    def save_checkpoint(self, path: str, model: torch.nn.Module) -> None:
        path: PurePath = PurePath(path)
        Path(path.parent).mkdir(parents=True, exist_ok=True)
        if self.last_model_path == None:
            self.last_model_path = str(path)
            print("get the self.last_model_path: ", self.last_model_path)

        ckpt = {
            "round": self.current_round,
            "global_steps": self.current_step,
            "model": model.state_dict(),
            "optimizer": self.opt_state,
            "scheduler": self.sch_state,
        }
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str, model: torch.nn.Module) -> torch.nn.Module:
        ckpt: dict = torch.load(path)

        self.current_step = ckpt.get("global_steps", 0)
        self.current_round = ckpt.get("round", 0)
        self.opt_state = ckpt.get("optimizer", None)
        self.sch_state = ckpt.get("scheduler", None)

        model.load_state_dict(ckpt["model"])
        return model
    
    def cleanup(self):
        self.opt_state = deepcopy(self.opt.state_dict())
        self.sch = None
        self.opt = None
        self.model = None
        self.global_model = None

        self.logger = None
        self.abort_signal = None

        torch.cuda.empty_cache()

    def setup(self, model: torch.nn.Module, logger: SummaryWriter,abort_signal: Any):
        torch.cuda.empty_cache()
        self.model = model
        self.global_model: torch.nn.Module = get_model(self.model_config)
        self.global_model.load_state_dict(deepcopy(model.state_dict()),strict=False)
        self.global_model.eval()

        self.logger = logger
        if abort_signal is not None:
            self.abort = abort_signal
        else:
            self.abort = None

        self.configure_optimizer()

    def configure_optimizer(self):
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        if self.opt_state is not None:
            try:
                self.opt.load_state_dict(self.opt_state)
            except:
                print("first lora initiated, no opt parameter loaded")

    def training_loop(self, data_loader: DataLoader, num_steps: int, device: str = "cuda"):
        self.model = self.model.to(device)
        
        target_step = self.current_step + num_steps
        total_loss = 0.0
        total_batches = 0

        for batch in self.get_batch(data_loader,num_steps):
            with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                loss = self.training_step(self.model, batch)

                total_loss += loss.item()
                total_batches += 1

            self.opt.zero_grad()

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2.0)

            # Apply gradient
            self.scaler.step(self.opt)
            # self.sch.step()
            self.scaler.update()

            self.current_step += 1
            if self.current_step >= target_step:
                break
            if self.abort is not None and self.abort.triggered:
                break

        avg_loss = total_loss / total_batches
        return avg_loss

    def get_batch(self, data_loader: DataLoader, num_steps: int):
        it = iter(data_loader)
        for i in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)
            yield batch

    def training_step(self, model: torch.nn.Module, batch: Dict, device: str = "cuda"):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        preds_new = model(image)
        loss = self.loss_fun(preds_new,label.to(torch.uint8))
        
        if self.last_model_path is not None:
            model_old = copy.deepcopy(model)
            model_old.load_state_dict(torch.load(self.last_model_path,weights_only=True)['model'])
            preds_old = model_old(image)
            
            lkd, kd_coef = self.lkd_loss(preds_new, preds_old)

            self.logger.add_scalar("lkd_loss loss", lkd, self.current_step)
            self.logger.add_scalar("kd_coef", kd_coef, self.current_step)
            self.logger.add_scalar("cross entropy loss", loss, self.current_step)
            
            loss = 1 * kd_coef * lkd + loss

            del model_old, preds_old, preds_new
            gc.collect()
            torch.cuda.empty_cache()

        if self.logger is not None:
            step = self.current_step
            self.logger.add_scalar("train loss", loss, step)
        
        return loss

    def get_loss(self):
        if self.dataset_info.dataset_jsons[self.dataset_id].get("is_classification_dataset", False):
            return ClassificationLoss(self.dataset_info.label_managers[self.dataset_id].has_regions)
        if self.dataset_info.label_managers[self.dataset_id].has_regions:
            loss = DC_and_BCE_loss({},
                                    {'batch_dice': self.dataset_info.configuration_managers[self.dataset_id].batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': False},
                                    use_ignore_label=self.dataset_info.label_managers[self.dataset_id].ignore_label is not None,
                                    dice_class=MemoryEfficientSoftDiceLoss)
        else:

            loss = DC_and_CE_loss({'batch_dice': self.dataset_info.configuration_managers[self.dataset_id].batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=self.dataset_info.label_managers[self.dataset_id].ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        return loss