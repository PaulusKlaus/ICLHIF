import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import data_util_pal

def plot_label_distribution(labels, title="Label distribution"):
    counts = Counter(labels.tolist() if isinstance(labels, np.ndarray) else labels)
    labs = sorted(counts.keys())
    vals = [counts[k] for k in labs]
    plt.figure()
    plt.bar([str(k) for k in labs], vals)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_time_signals(data, labels, samples_per_label=3, title="Time-domain samples"):
    """
    data: shape (N, 1024, 1) or (N, 1024)
    labels: shape (N,)
    """
    x = np.squeeze(data)  # (N, 1024)
    unique_labels = sorted(np.unique(labels))
    t = np.arange(x.shape[1])

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        sel = idx[:min(samples_per_label, len(idx))]
        plt.figure()
        for i in sel:
            plt.plot(t, x[i], alpha=0.8, label=f"Sample {i}", linewidth=1)
        plt.title(f"{title} – label {lab} (n={len(idx)})")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_spectra(data, labels, fs=None, samples_per_label=3, title="Amplitude spectrum"):
    """
    If your data are already Fourier-transformed (is_oneD_Fourier=True), this
    will just plot them. If they’re time-domain, we FFT them here.
    fs: optional sampling frequency to label the x-axis in Hz; otherwise bins.
    """
    x = np.squeeze(data)  # (N, 1024)
    N = x.shape[1]

    # If looks like time-domain (both +/− values), take FFT magnitude
    # Otherwise assume it's already |FFT|
    def ensure_fft_magnitude(arr):
        # heuristic: if negative values exist, compute FFT magnitude
        return np.abs(np.fft.fft(arr)) if np.any(arr < 0) else arr

    X = np.apply_along_axis(ensure_fft_magnitude, 1, x)
    # Take one-sided spectrum
    half = N // 2
    X = X[:, :half]
    if fs is not None:
        freqs = np.linspace(0, fs/2, half, endpoint=False)
        xlab = "Frequency (Hz)"
        xticks = freqs
    else:
        xlab = "Frequency bin"
        xticks = np.arange(half)

    unique_labels = sorted(np.unique(labels))
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        sel = idx[:min(samples_per_label, len(idx))]
        plt.figure()
        for i in sel:
            plt.plot(xticks, X[i], alpha=0.8, label=f"Sample {i}", linewidth=1)
        plt.title(f"{title} – label {lab}")
        plt.xlabel(xlab)
        plt.ylabel("|X(f)|")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example: pick labels you care about
data, labels = data_util_pal.get_Data_By_Label(
    mathandler=data_util_pal.MatHandler(is_oneD_Fourier=False),  # set True if you want pre-FFT
    pattern='full',                               # 'train' | 'val' | 'test' | other = train+val
    label_list=[]                           # include label 0 if you want “normal”
)

plot_label_distribution(labels)
plot_time_signals(data, labels, samples_per_label=3)

# If you want spectra (set fs if you know the sampling freq, e.g., fs=12000)
plot_spectra(data, labels, fs=None, samples_per_label=3)


ds = data_util_pal.load_Dataset_Original(label_list=[0,1,2,3], batch_size=32, is_oneD_Fourier=False, pattern='train')

# Take one batch for quick plots
for batch in ds.take(1):
    batch_np = batch.numpy() if hasattr(batch, "numpy") else np.array(batch)
    # No labels in this dataset pipeline, so just fake a single label for the batch:
    fake_labels = np.zeros((batch_np.shape[0],), dtype=int)
    plot_time_signals(batch_np, fake_labels, samples_per_label=5)
    plot_spectra(batch_np, fake_labels, fs=None, samples_per_label=5)
