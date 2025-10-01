import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from typing import Optional, Iterable, Tuple, List 
from os import path, makedirs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from builder import SimSiam, BackBone1D, Projector_MLP, oneD_Fourier_view
from data_glue import build_loaders_from_mat



# Adjust `sys.argv` for compatibility with Jupyter Notebook or IPython environments.
if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = [sys.argv[0]]  # Reset `sys.argv` to prevent parsing issues.
 
# Define configuration parameters for the SimSiam experiment using argparse.Namespace.
args = argparse.Namespace(
    data_root='./data',          # Path to the root directory containing dataset.
    exp_dir='./experiments',     # Directory for saving experimental results (e.g., checkpoints, logs).
    trial='1',                   # Identifier for the experiment trial.
    in_dim=1,   # or 1024                # Dimension of the input images (e.g., 32x32 for CIFAR-10).
    #arch='resnet18',             # Backbone architecture to use (e.g., ResNet18).
    feat_dim=16,               # Dimensionality of the projected features.
    #num_proj_layers=2,           # Number of layers in the projection MLP.
    batch_size=16,              # Batch size for training and validation.
    num_workers=1,               # Number of data loading workers.
    epochs=100,                  # Number of training epochs.
    gpu=0,                       # GPU index to use for training (e.g., 0 for the first GPU).
    loss_version='simple',   # Version of the loss function ('simplified' or 'original').
    print_freq=10,               # Frequency (in batches) to print training progress.
    eval_freq=5,                 # Frequency (in epochs) to perform KNN evaluation.
    save_freq=50,                # Frequency (in epochs) to save model checkpoints.
    resume=None,                 # Path to a checkpoint file to resume training, if any.
    learning_rate=0.06,          # Initial learning rate for the optimizer.
    weight_decay=5e-4,           # Weight decay for regularization.
    momentum=0.9                 # Momentum for the SGD optimizer.
)
print("Parsed Arguments:", args)



def set_visible_gpus(gpus: str = "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def move_to(x, device):
    if isinstance(x, (list, tuple)):
        return [move_to(t, device) for t in x]
    return x.to(device, non_blocking=True)


# Training step 

def train_step(
    model: nn.Module,
    x_time: torch.Tensor,
    x_freq: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: Optional[float] = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if scaler is None:
        out = model(x_time, x_freq)
        loss = out["loss"]
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        return float(loss.item())
    else:
        with torch.amp.autocast('cuda'):
            out = model(x_time, x_freq)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.item())    



# Epoch Loop 
# Since we dont need to savew the gradient during the validation 
@torch.no_grad()
def validate_epoch(
    model : nn.Module,
    loader_time : DataLoader,
    loader_freq : DataLoader,
    device : torch.device
) -> float:
    
    model.eval()
    losses = []

    for(xt,), (xf,) in zip(loader_time, loader_freq):
        xt = move_to(xt, device)
        xf = move_to(xf, device)
        loss = model(xt, xf)["loss"]
        losses.append(float(loss.item()))
    return sum(losses)/ max(1, len(losses))



def train_CL(
    model: nn.Module,
    loader_time: DataLoader,
    loader_freq: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 100,
    device: Optional[torch.device] = None,
    amp: bool = False,
    log_every: int = 2,
    val_loaders: Optional[Tuple[DataLoader, DataLoader]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Mirrors the TF `train_CL`. Returns (epoch_train_losses, epoch_val_losses).
    Assumes the two loaders yield tensors with shapes (B, 1, N) and are zipped in lockstep.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    epoch_train_losses, epoch_val_losses = [], []

    steps_per_epoch = min(len(loader_time), len(loader_freq))
    for epoch in range(1, epochs + 1):
        step_losses = []
        it_time = iter(loader_time)
        it_freq = iter(loader_freq)

        for _ in range(steps_per_epoch):
            xt = next(it_time)[0]  # expect loader to return (tensor,) or (tensor, label). We take first.
            xf = next(it_freq)[0]
            xt = move_to(xt, device)
            xf = move_to(xf, device)

            loss_val = train_step(model, xt, xf, optimizer, scaler)
            step_losses.append(loss_val)

            if scheduler and not isinstance(scheduler, CosineAnnealingLR):
                # for per-step schedulers (rare), step here
                scheduler.step()

        epoch_loss = sum(step_losses) / max(1, len(step_losses))
        epoch_train_losses.append(epoch_loss)

        # epoch-wise scheduler step (CosineAnnealingLR typical)
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()

        # optional validation
        if val_loaders is not None:
            v_time, v_freq = val_loaders
            val_loss = validate_epoch(model, v_time, v_freq, device)
            epoch_val_losses.append(val_loss)
        else:
            val_loss = None
            epoch_val_losses.append(val_loss)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            if val_loss is not None:
                print(f"[Epoch {epoch:03d}] train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"[Epoch {epoch:03d}] train_loss={epoch_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    return epoch_train_losses, epoch_val_losses


# Checkpoint helpers 

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    print(f"Loaded checkpoint from {path}")

# ----------------------------
# A "Step_Original" analogue
# ----------------------------

def Step_Original(
    output_dir: str,
    save_model_name: str,
    epochs: int,
    batch_size: int,
    label_list: List[int],
    load_model_path: str = "",
    save_predictor_name: str = None,  # (optional) if you want to stash predictor separately
    gpus: str = "0",
):
    """
    PyTorch version of TF Step_Original.

    Expects TWO DataLoaders, one for time-domain, one for frequency-domain,
    zipped in lockstep. Replace the dataset creation block with your own.
    """
    set_visible_gpus(gpus)


    # ---------- Build datasets ----------
    # TODO: replace with your actual dataset creation. We expect each loader
    # to yield batches shaped (B, 1, N). If you already have only time-series,
    # you can compute FFT on-the-fly in the loop instead of making a second loader.
    #
    # Example skeleton:
    #
    # dataset_time = MyDataset(label_list=label_list, split="train", transform=None)
    # dataset_freq = MyDataset(label_list=label_list, split="train", transform=oneD_Fourier_view)
    # loader_time = DataLoader(dataset_time, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    # loader_freq = DataLoader(dataset_freq, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    loader_time, loader_freq = build_loaders_from_mat(batch_size=16,
                                                      pattern="train",
                                                      num_workers=4,
                                                      pin_memory=True)
    
    
    raise_if_not_replaced = False
    if raise_if_not_replaced:
        raise RuntimeError("Replace the dataset creation block in Step_Original with your own DataLoaders.")

    # ---------- Build model ----------
    backbone = BackBone1D(in_channel=1, out_channel=256)
    projector = Projector_MLP(in_dim=4096, hidden_dim=256, out_dim=16)
    model = SimSiam(in_channel=1, out_channel=16, backbone=backbone, projector=projector)

    # ---------- Optimizer & scheduler (cosine) ----------
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # cosine over total epochs (epoch-wise step to match TF's CosineDecay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    # ---------- Optional resume ----------
    if load_model_path and os.path.isfile(load_model_path):
        load_checkpoint(model, optimizer, load_model_path, scheduler=scheduler, map_location="cpu")

    # ---------- Train ----------
    train_losses, _ = train_CL(
        model=model,
        loader_time=loader_time,
        loader_freq=loader_freq,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        amp=True,          # toggle if you like
        log_every=2,
        val_loaders=None,  # or pass (val_time_loader, val_freq_loader)
    )

    # ---------- Save ----------
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = path.join(output_dir, save_model_name.replace(".h5", ".pt"))
    save_checkpoint(model, optimizer, ckpt_path, scheduler=scheduler)

    # (Optional) save predictor separately
    if save_predictor_name:
        pred_only_path = path.join(output_dir, save_predictor_name.replace(".h5", ".pt"))
        torch.save({"predictor": model.predictor.state_dict()}, pred_only_path)
        print(f"Saved predictor weights to {pred_only_path}")

    # ---------- Plot & save curve ----------
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    fig_path = path.join(output_dir, "epoch_wise_loss.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved loss curve to {fig_path}")

# ----------------------------
# Multi-step runner (like TF)
# ----------------------------

def train_Step(step: str):
    if step == "one":
        Step_Original(
            epochs=50,
            output_dir="experiments/Step_One",
            save_model_name="Step_One_147.h5",
            label_list=[],
            save_predictor_name="Step_One_Predictor_147.h5",
            batch_size=10,
            gpus="0",
        )
    elif step == "two":
        Step_Original(
            epochs=50,
            output_dir="experiments/Step_Two",
            save_model_name="Step_Two_147.h5",
            label_list=[1],
            save_predictor_name="Step_Two_Predictor_147.h5",
            batch_size=10,
            gpus="0",
            load_model_path="experiments/Step_One/Step_One_147.pt",  # resume from step one
        )
    elif step == "three":
        Step_Original(
            epochs=50,
            output_dir="experiments/Step_Three",
            save_model_name="Step_Three_147.h5",
            label_list=[1, 4],
            save_predictor_name="Step_Three_Predictor_147.h5",
            batch_size=10,
            gpus="0",
            load_model_path="experiments/Step_Two/Step_Two_147.pt",
        )
    print("suc")

if __name__ == "__main__":
    train_Step("one")

