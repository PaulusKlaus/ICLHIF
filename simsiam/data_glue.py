# data_glue.py
import torch
from torch.utils.data import Dataset, DataLoader

from data_util_pal import MatHandler  # where your MatHandler lives (the code you pasted)

class NumpyTensorDataset(Dataset):
    """Wrap a numpy array of shape (N, 1, 1024) into a float32 tensor dataset."""
    def __init__(self, np_array):
        assert np_array.ndim == 3 and np_array.shape[1] == 1, f"Expected (N,1,1024), got {np_array.shape}"
        # IMPORTANT: float32 to match model parameters (your code creates float64 by default)
        self.tensor = torch.from_numpy(np_array).float()

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        # return a 3D tensor (1, 1024) inside a tuple so training loop can do (xt,) pattern
        return (self.tensor[idx],)

def build_loaders_from_mat(
    batch_size: int = 16,
    pattern: str = "train",
    label_list = [],
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Returns: loader_time, loader_freq
    - time loader: MatHandler(..., is_oneD_Fourier=False)
    - freq loader: MatHandler(..., is_oneD_Fourier=True)
    """
    mh_time = MatHandler(is_oneD_Fourier=False)
    mh_freq = MatHandler(is_oneD_Fourier=True)

    # pick a split
    if pattern == "train":
        X_t = mh_time.X_train
        X_f = mh_freq.X_train
    elif pattern == "val":
        X_t = mh_time.X_val
        X_f = mh_freq.X_val
    elif pattern == "test":
        X_t = mh_time.X_test
        X_f = mh_freq.X_test
    else:
        raise ValueError("pattern must be 'train'|'val'|'test'")

    # Optionally filter by labels (exactly like your get_Data_By_Label does).
    # If you need that behavior, reuse get_Data_By_Label instead of direct arrays.

    ds_time = NumpyTensorDataset(X_t)  # (N,1,1024) -> float32
    ds_freq = NumpyTensorDataset(X_f)

    # drop_last=True is important so zip() keeps them in lockstep for SimSiam
    loader_time = DataLoader(ds_time, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    loader_freq = DataLoader(ds_freq, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    return loader_time, loader_freq
