import os
import json
import torch
import csv
from loguru import logger
import numpy as np
from tqdm import tqdm
import itertools
import sys
import argparse
sys.path.append("src")
from src.data.data_manager import DataManager
from src.trainer._trainer import Trainer
from src.utils.get_model import get_model
from src.utils.model_weights import extract_weights, load_weights, extract_encoder_weights
from nnunetv2_utils.default_preprocessor import CustomizedPreprocessor, nnUNetComponentUtils
from nnunetv2_utils.data_loader import CustomizednnUNetDataLoader3D
from monai.inferers import sliding_window_inference
from src.utils.model_weights import load_weights
from typing import List, Dict, Callable
from sklearn.metrics import confusion_matrix
# from utils.lora import apply_lora_to_conv3d
from src.utils.model_weights import load_weights, extract_encoder_weights
import pandas as pd
import multiprocessing
import nibabel as nib
import torch.nn.functional as F

def save_to_csv(list_a,dict_labels,csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        header = ['name'] + list(dict_labels.keys())
        writer.writerow(header)

        # Write the rows
        for i in range(len(list_a)):
            # row = [list_a[i]] + [dict_labels[label][i].numpy() for label in dict_labels]
            row = [list_a[i]] + [
                dict_labels[label][i].numpy() if hasattr(dict_labels[label][i], 'numpy') else dict_labels[label][i]
                for label in dict_labels
            ]
            writer.writerow(row)

def restore_image(image, properties):
    """
    image: torch.Tensor [C, X, Y, Z]
    properties: dict from preprocess_online
    """
    # Step 1: Resample back to original shape before cropping
    original_spacing = properties['sitk_stuff']['spacing']
    resampled_spacing = properties['spacing']

    shape_after_crop = properties['shape_after_cropping_and_before_resampling']  # (144, 174, 147)
    shape_before_crop = properties['shape_before_cropping']  # (155, 240, 240)

    # Calculate scaling factors
    scale_factors = [o / a for o, a in zip(shape_after_crop, image.shape[1:])]
    
    # Use F.interpolate to upsample back to shape_after_cropping_and_before_resampling
    image = F.interpolate(
        image.unsqueeze(0),  # add batch dim
        size=shape_after_crop,
        mode='trilinear',
        align_corners=False
    ).squeeze(0)

    # Step 2: Pad back to shape_before_cropping
    bbox = properties['bbox_used_for_cropping']
    target_shape = shape_before_crop

    # Calculate paddings
    pad_x = [bbox[0][0], target_shape[0] - bbox[0][1]]
    pad_y = [bbox[1][0], target_shape[1] - bbox[1][1]]
    pad_z = [bbox[2][0], target_shape[2] - bbox[2][1]]

    # torch.nn.functional.pad expects pad in reverse order: (z2, z1, y2, y1, x2, x1)
    padding = [pad_z[0], pad_z[1], pad_y[0], pad_y[1], pad_x[0], pad_x[1]]

    image = F.pad(image, padding)

    return image

def save_modalities_to_nii(restored_image, properties, output_dir, basename="modality"):
    # os.makedirs(output_dir, exist_ok=True)

    spacing = properties['sitk_stuff']['spacing']
    origin = properties['sitk_stuff']['origin']
    direction = properties['sitk_stuff']['direction']

    # Convert direction to 3x3 matrix if needed
    direction = np.array(direction).reshape(3, 3)

    # Loop over each modality/channel
    for i in range(restored_image.shape[0]):
        modality_data = restored_image[i].cpu().numpy()

        # Create affine
        affine = np.eye(4)
        affine[:3, :3] = np.diag(spacing) @ direction
        affine[:3, 3] = origin

        # Create NIfTI image
        nii_img = nib.Nifti1Image(modality_data, affine)

        # Save
        base_path = output_dir[i].replace('_raw', '_results')
        nib.save(nii_img, base_path)

        print(f"Saved modality {i} to {base_path}")

class AugmentedPredictor:
    def __init__(self, predictor: Callable):
        """
            Mimic nnUNetv2's test-time augmentation

            Reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py#L537
        """
        self.predictor = predictor
    
    def __call__(self, x: torch.Tensor, *args, **kwargs):
        pred = self.predictor(x, *args, **kwargs)

        mirror_axes = list(range(len(x.shape) - 2))
        mirror_axes = [m + 2 for m in mirror_axes]
        axes_combinations = [c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)]

        for axes in axes_combinations: pred += torch.flip(self.predictor(torch.flip(x, axes), *args, **kwargs), axes)

        return pred / float(len(axes_combinations) + 1)

def filter_label(x: torch.Tensor, cls_idx):
    ret = torch.zeros_like(x).to(bool)
    if isinstance(cls_idx, list):
        for idx in cls_idx:
            ret[x == idx] = 1
    else: 
        ret[x == cls_idx] = 1
    return ret

def eval_cls(pred: torch.Tensor, label: torch.Tensor, cls_idx: List[int]):
    pred_filtered = filter_label(pred, cls_idx)
    label_filtered = filter_label(label, cls_idx)
    if torch.sum(label_filtered) == 0: return 1.
    intersection = pred_filtered & label_filtered
    return 2 * torch.sum(intersection) / (torch.sum(pred_filtered) + torch.sum(label_filtered))



class evaluat_cross_site():
    def __init__(self,workspace_folder,client_list,gpu_id):
        self.sim_folder = os.path.join(workspace_folder, "simulate_job")
        self.client_list=client_list
        self.device =  torch.device(f"cuda:{gpu_id}")
        self.workspace_folder = workspace_folder
    def get_weights(self, ch_paths):
        return torch.load(ch_paths)["model"]

    def run(self):
        try:
            glob_weights = self.get_weights(os.path.join(self.sim_folder, "app_server", "best_FL_global_model.pt"))
        except:
            glob_weights = None
        # for client_folder in os.listdir(self.sim_folder):
        for client_folder in self.client_list:
            if client_folder in ["app_server", "pool_stats", "cross_site_val"] or "." in client_folder: continue
            config_file = os.path.join(self.sim_folder, client_folder, "config", "config_task.json")
            with open(config_file, "r") as f: self.CFGS = json.load(f)
           
            # logger.info(self.dataset_name)
            dataset_id = self.CFGS["dataset_id"]
            self.dataset_info = nnUNetComponentUtils([dataset_id], self.CFGS["path_config"])
            self.client_model=get_model(self.CFGS["model"])
            model_path = os.path.join(self.sim_folder, client_folder, "models", "best_model.pt")
            # model_path = os.path.join(self.sim_folder, client_folder, "models", "last.pt")
            logger.info(model_path)

            # apply_lora_to_conv3d(unet_model=self.client_model)

            # load client model and weights 
            logger.info(model_path)
            self.client_model.load_state_dict(torch.load(model_path)["model"])

            # load global encoder weights
            if glob_weights!=None:
                self.client_model=load_weights(self.client_model, glob_weights)
            # logger.info(glob_weights.keys())
            # choose the testor
            self.dataset_info.dataset_jsons[self.dataset_info.dataset_ids[0]].get("is_classification_dataset", False) 
            if self.dataset_info.dataset_jsons[dataset_id].get("is_classification_dataset", False):
                logger.info("use cls_testor")
                self.testor=Cls_Testor(self.dataset_info,self.CFGS["notes"],self.device, self.workspace_folder)
            else:
                logger.info("use testor")
                self.testor=Testor(self.dataset_info,self.CFGS["notes"],self.device,self.workspace_folder)

            self.preprocessor = CustomizedPreprocessor(path_conf=self.CFGS["path_config"], verbose=True)
            data_list=self.get_data_list(dataset_id)

            self.testor.run_actual(self.client_model.to(self.device),data_list,self.preprocessor)
    

    def get_test_identifiers(self, folder: str) -> List[str]:
        """
        finds all nii.gz files in the given folder and reconstructs the training case names from them
        """
        # case_identifiers = [os.path.join(folder,i) for i in os.listdir(folder)]
        case_identifiers = [os.path.join(folder, i) for i in os.listdir(folder) if '_0000.nii.gz' in i]
        return case_identifiers

    def separate_data_list_modal(self,data):
        data_list = []
        for img in data['image']:
            base_name=os.path.basename(img)
            dir_name=os.path.dirname(img)
            key = "_".join(base_name.split("_")[:-1])
            label_path=os.path.dirname(dir_name)+'/labelsTs/'+ key+'.nii.gz'
            image_pref=os.path.dirname(dir_name)+'/imagesTs/'+ key
            dataset_id=data['dataset_id']
            modalities=[image_pref+'_0000.nii.gz', image_pref+'_0001.nii.gz', image_pref+'_0002.nii.gz', image_pref+'_0003.nii.gz']
            img_list=[]
            for idx, _ in enumerate(self.dataset_info.dataset_jsons[dataset_id]["channel_names"]):
                img_list.append(modalities[idx])
            data_list.append({"image": img_list, "label": label_path, "dataset_id": dataset_id})
        return data_list

    def get_data_list(self,dataset_id):
        self.dataset_info.raw_folders[dataset_id]
        test_identifiers = self.get_test_identifiers(os.path.join(self.dataset_info.raw_folders[dataset_id], "imagesTs"))
        label_path=self.get_test_identifiers(os.path.join(self.dataset_info.raw_folders[dataset_id], "labelsTs"))
        data_list=[{"image": test_identifiers,"label":label_path,"dataset_id":dataset_id}]
        data_list_modal=self.separate_data_list_modal(data_list[0])

        return data_list_modal

class Testor():
    def __init__(self, dataset_info: nnUNetComponentUtils, dataset_name,device,workspace_folder) -> None:
        # `dataset_info` contains always 1 single dataset and hence 50 steps is enough
        self.dataset_info = dataset_info
        self.dataset_id = dataset_info.dataset_ids
        self.dataset_name = dataset_name
        self.device = device
        self.workspace = workspace_folder


    @torch.no_grad()
    def batch_label_map_to_cls_label(self, label_map: torch.Tensor, num_classes: int):
        label_map_one_hot: torch.Tensor = torch.nn.functional.one_hot(label_map.to(torch.int64).squeeze(1), num_classes=num_classes + 1)
        spatial_dims = list(range(1, len(label_map_one_hot.shape) - 1))
        label_one_hot = torch.sum(label_map_one_hot, dim=spatial_dims)
        label_one_hot = label_one_hot[:, 1:] # exclude background; in classification, we never have a channel for background
        return torch.argmax(label_one_hot, dim=1)

    @torch.no_grad()
    def run_actual(self, model,data_list, preprocessor:CustomizedPreprocessor):
        """
            'data_list':[{"image": ["paths"],"label":"path","dataset_id":ID}]
            please put the same modalities of a data in the ["paths"]
        """
        
        model.eval()
        # # first collect the files according to IDs data_list_by_id = {k:[] for k in self.dataset_info.dataset ids}for data item in data list:
        data_list_by_id = {k:[] for k in self.dataset_info.dataset_ids}
        for data_item in data_list:
            dataset_id = data_item["dataset_id"]
            if dataset_id not in data_list_by_id:
                raise ValueError(f"Dataset ID {dataset_id} not supported here. Supported: {self.dataset_info.dataset_ids}")
            data_list_by_id[dataset_id].append(data_item)

        # then evaluate by dataset IDs
        all_res =[]

        for k in data_list_by_id:
            self.patch_size = self.dataset_info.configuration_managers[k].patch_size
            self.batch_size= self.dataset_info.configuration_managers[k].batch_size
            logger.info(self.patch_size)
            logger.info(self.batch_size)
            rw=self.dataset_info.plans_managers[k].image_reader_writer_class()
            labels_dict, preds, labels, eval_name=self.eval_step(data_list_by_id,k,preprocessor,dataset_id,model,rw)
            tmp_res: Dict[str, torch.Tensor]=self.eval_res(labels_dict, preds, labels,eval_name)
            logger.info(tmp_res)
            
            tmps_res=self.compute_res(tmp_res)
            all_res.append(tmps_res)

        print(all_res, flush=True)
        with open(f'{self.workspace}/{self.dataset_id}_{self.dataset_name}.txt', 'a') as file:
            for items in data_list_by_id:
                file.write(f"{items}\n")
            for items in all_res:
                file.write(f"{items}\n")
        print(f'{self.workspace}/{self.dataset_id}_{self.dataset_name}.txt saved')
        
        # Saving to CSV
        df = pd.DataFrame(all_res)
        csv_file_path = f'{self.workspace}/{self.dataset_id}_{self.dataset_name}.csv'
        print(f"{csv_file_path} saved")
        df.to_csv(csv_file_path, index=False)

    def compute_res(self, tmp_res):
        return {k:tmp_res[k].item()for k in tmp_res}



    def eval_step(self,data_list_by_id,k,preprocessor,dataset_id,model,rw):
        preds, labels, eval_name =[],[], []
        for data_item in data_list_by_id[k]:
            # You shall never see the labels before evaluatior
            image,_,properties = preprocessor.preprocess_online(data_item["image"], None, k)
            image = torch.from_numpy(image).to(self.device)
            # logger.info(image.element_size() * image.numel()/(1024 ** 3))
            prediction: torch.Tensor = sliding_window_inference(image.unsqueeze(0), self.patch_size, 1, AugmentedPredictor(model), mode="constant", overlap=0.5,device='cpu').squeeze(0)
            # prediction: torch.Tensor = sliding_window_inference(image.unsqueeze(0), patch_size, slef.batch_size * 3, AugmentedPredictor(model), mode="gaussian", overlap=0.5).squeeze(0)
            inverted_pred=preprocessor.inverse_preprocess_online(prediction.squeeze(0).detach().cpu().numpy(), properties,k)
            label, label_ref_dict = rw.read_seg(data_item["label"])
            preds.append(torch.from_numpy(inverted_pred))
            labels.append(torch.from_numpy(label.squeeze(0)))
            eval_name.append(os.path.basename(data_item['image'][0]).replace('_0000.nii.gz',''))

            if 0: 
                # save the results
                label_path = data_item["image"][0].replace('_raw', '_results').replace('_0000', '_label')
                pred_path = data_item["image"][0].replace('_raw', '_results').replace('_0000', '_pred')

                # Ensure directories exist
                for path in [label_path, pred_path]:
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                # Save NIfTI images
                nib.save(nib.Nifti1Image(label.squeeze(0), affine=np.eye(4)), label_path)
                nib.save(nib.Nifti1Image(inverted_pred, affine=np.eye(4)), pred_path)

                image_save = restore_image(image, properties)
                save_modalities_to_nii(image_save, properties, data_item["image"])
                # for i in np.arange(image.shape[0]):
                    
                #     image_save=image[i,:,:,:].cpu().numpy()
                #     print("image_save: ", image_save.shape)
                #     print("properties: ", properties)
                #     # inverse_image = preprocessor.inverse_preprocess_online(image.detach().cpu().numpy(), properties,k)
                #     print("inverse_image: ", inverse_image.shape)

            
            # print("image: ", image.shape)
            # print("label: ", label.shape)
            # print("prediction: ", prediction.shape)
            # print("inverted_pred: ", inverted_pred.shape)

            # exit()
        labels_dict = self.dataset_info.dataset_jsons[k]["labels"] 
            # logger.info(dice_score(inverted_pred, label, num_classes=int(np.max(label))))
        return labels_dict, preds, labels, eval_name

    def eval_res(self, label_dict, preds, labels,eval_names):
        res_dict = {}
        res_dict_all={}

        for label, pred in zip(labels, preds):
            for label_name in label_dict:
                if label_name == "background": continue
                if label_name not in res_dict: res_dict[label_name] = 0.
                if label_name not in res_dict_all: res_dict_all[label_name] = []
                dice = eval_cls(pred, label, label_dict[label_name])
                print(label_name, dice, flush=True)
                res_dict[label_name] += dice
                res_dict_all[label_name].append(dice)
        
        avg_dict = {label_name: res_dict[label_name] / len(labels) for label_name in res_dict}
        csv_file_path = f'{self.workspace}/{self.dataset_id}_{self.dataset_name}_all.csv'
        save_to_csv(eval_names,res_dict_all,csv_file_path)
        print("avg", avg_dict, flush=True)
        print("cls avg", np.mean([avg_dict[k] for k in avg_dict]), flush=True)
        return avg_dict

class Cls_Testor(Testor):
    def __init__(self, dataset_info: nnUNetComponentUtils,dataset_name, device, workspace) -> None:
        super().__init__(dataset_info,dataset_name,device,workspace)
        print("!!!!!USING CLASSIFICATION Testor!!!!")
        self.r=5.0
        self.device = device

    def lse(self, prediction_map):
        exp_x = torch.exp((self.r * prediction_map).clamp(max=50)) 
        result = torch.log(torch.mean(exp_x, dim=list(range(2, len(prediction_map.shape)))).clamp(min=1e-6)) / self.r
        return result

    def eval_step(self,data_list_by_id,k,preprocessor,dataset_id,model,rw):
        preds, labels, eval_name =[],[], []
        for data_item in tqdm(data_list_by_id[k], desc=f"val {k}"):
            # You shall never see the labels before evaluatior
            image,_,properties = preprocessor.preprocess_online(data_item["image"], None, k)
            image = torch.from_numpy(image).to(self.device)
            # logger.info(image.shape)
            logger.info(image.element_size() * image.numel()/(1024 ** 3))
            prediction: torch.Tensor = sliding_window_inference(image.unsqueeze(0), self.patch_size, 1, AugmentedPredictor(model), mode="gaussian", overlap=0.5,device='cpu')
            # prediction: torch.Tensor = sliding_window_inference(image.unsqueeze(0), patch_size, slef.batch_size * 3, AugmentedPredictor(model), mode="gaussian", overlap=0.5)
            # set batchsize for 1:batch_size×channel_size×D×H×W,
            predicted_logits = self.lse(prediction)
            # inverted_pred=preprocessor.inverse_preprocess_online(prediction.squeeze(0).detach().cpu().numpy(), properties,k)
            label, label_ref_dict = rw.read_seg(data_item["label"])
            label = self.batch_label_map_to_cls_label(torch.from_numpy(label), predicted_logits.shape[1])
            preds.append(torch.argmax(predicted_logits))
            labels.append(label.squeeze(0))
            eval_name.append(os.path.basename(data_item['image'][0]).replace('_0000.nii.gz',''))

            
        labels_dict = self.dataset_info.dataset_jsons[k]["labels"] 
        return labels_dict, preds, labels, eval_name

    def compute_res(self, tmp_res):
        return tmp_res

    def confusion_mat_to_acc(self, mat: np.ndarray):
        """
            by acc, we mean: torch.sum(pred == label) / label.Size()
        """
        return np.trace(mat) / np.sum(mat)

    def eval_res(self, label_dict, preds, labels,eval_name):
        # logger.info(label_dict)
        # logger.info(len(preds))
        res_dict = {}
        # np.savez(f'results/{self.dataset_name}_test.npz', var1=label_dict, var2=preds, var3=labels)
        confusion_mat = confusion_matrix(labels, preds)
        # accs = [confusion_mat_to_acc(confusion_mat[k]) for k in self.dataset_info.dataset_ids]
        accs =  self.confusion_mat_to_acc(confusion_mat)
        # print(f'results/{self.dataset_name}_test.npz saved')


        return accs

def worker(workspace, clients, gpu_id):
    eval = evaluat_cross_site(workspace, clients, gpu_id)
    eval.run()

def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-site data.")
    parser.add_argument('--workspace', type=str, default="workspace/partial_label_1",
                        help='Path to the workspace folder')
    parser.add_argument('--clients', type=str, nargs='+', default=['app_client3','app_client41','app_client5','app_client7'],
                        help='List of clients to evaluate')
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0,1,2,3], help="List of GPU IDs to use for each client.")
    args = parser.parse_args()


    processes = []
    for i, client in enumerate(args.clients):
        p = multiprocessing.Process(target=worker, args=(args.workspace, [client], i))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()

# example script:
# python test_parallel.py --workspace workspace/8c_L_Lu_HP --clients  app_client41 app_client42 app_client8 app_client666    --gpu_ids 0 1 2 3
# python test_parallel.py --workspace workspace/6client --clients app_client666 app_client42 app_client41 app_client3 --gpu_ids 0 1 2 3
# python test_parallel.py --workspace workspace/8c_L_HP --clients  app_client3 app_client5 app_client8 app_client777 --gpu_ids 0 1 2 3 
# python test_parallel.py --workspace workspace/8c_L_Lung_18brats --clients  app_client42 app_client41 app_client666 app_client777 --gpu_ids 0 1 2 3 