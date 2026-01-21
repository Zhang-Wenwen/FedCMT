import re
import multiprocessing
from time import sleep
from typing import Callable, List, Any
from multiprocessing.pool import Pool, ApplyResult
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np
from scipy.optimize import minimize
import torch

def apply_args_and_kwargs(func:Callable, args, kwargs):
    return func(*args, **kwargs)


def starmap_async_with_kwargs(pool:Pool, func:Callable, *args, **kwargs):
    return pool.starmap_async(apply_args_and_kwargs, [(func, args, kwargs)])


def spawn_processes(func:Callable, datalist:List[Any], num_processes:int, verbose:bool, desc:str=None, *args, **kwargs):
    """
        For functions that the first argument is an element from `datalist` to handle in multi-process

        References: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/preprocessors/default_preprocessor.py#L232
        
        Note that don't pass a Python closure or a lambda expression here, as they're not pickle-able
    """
    if num_processes in [0, 1]:
        ret: List[Any] = []
        for data_item in tqdm(datalist, disable=verbose, desc=desc):
            ret.append(func(data_item, *args, **kwargs))
        return ret
    ret:List[ApplyResult] = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        remaining = list(range(len(datalist)))
        workers = [i for i in p._pool]
        for data_item in datalist:
            ret.append(
                starmap_async_with_kwargs(p, func, data_item, *args, **kwargs)
            )
        with tqdm(total=len(datalist), disable=verbose, desc=desc) as pbar:
            while len(remaining) > 0:
                all_alive = all([i.is_alive() for i in workers])
                if not all_alive: raise RuntimeError(f"One of the background processes is missing")
                finished = [i for i in remaining if ret[i].ready()]
                _ = [ret[i].get() for i in finished] # get done so that errors can be raised
                for _ in finished:
                    ret[_].get() # allows triggering errors
                    pbar.update()
                remaining = [i for i in remaining if i not in finished]
                sleep(0.1)
    return [i.get()[0] for i in ret]


def get_client_id(client_name: str):
    try:
        client_id = int(client_name.replace("client", ""))
    except:
        num_str = re.sub(r"\D", "", client_name)
        client_id = int(num_str) if len(num_str) > 0 else 0
    return client_id


def pretty_results(res: dict):
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    for m in sorted(res.keys()):
        table.add_row([m, res[m]])
    return str(table)


def get_ca_delta(tensor_delta, alpha, rescale=1):
    """
    Solve for aggregated conflict-averse delta
    """
    shape_old = tensor_delta.shape
    tensor_delta = tensor_delta.reshape(1,-1)
    N = 1
    grads = tensor_delta.detach().t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {'type': 'eq', 'fun': lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)
    
    return final_update.reshape(shape_old)


def get_delta_tensor(param_tensor_new, param_tensor_old):
    """
    Calculate the difference between the current and last parameters (as tensors).

    Args:
        param_tensor (torch.Tensor): Current parameters as a tensor.
        last_param_tensor (torch.Tensor): Last parameters as a tensor.

    Returns:
        torch.Tensor: A tensor with the differences for all elements.
    """
    return param_tensor_new - param_tensor_old


def compute_likely_hood(x, delta):

    x = torch.clamp(x, min=1e-6, max=1-1e-6)
    delta = torch.clamp(delta, min=1e-6, max=1-1e-6)

    nll = - (x * torch.log(delta) + (1 - x) * torch.log(1 - delta))

    nll_max = torch.tensor(2.0)  
    normalized_nll = nll / nll_max  

    similarity_prob = 1 - normalized_nll


    return similarity_prob.mean().item()