# test_mat_handler.py
import os
import shutil
import numpy as np
import pytest
from pathlib import Path
from scipy.io import savemat

# Import the code under test
# If your file is not named mat_handler.py, change this import accordingly.
import data_util_pal as mh


# ---------- Helpers & Fixtures ----------

@pytest.fixture
def tmp_oneD_dir(tmp_path, monkeypatch):
    """
    Create a temporary working directory with a 'oneD' folder
    containing a few small .mat files. We chdir into it during tests
    so the module's os.listdir('oneD') works without modification.
    """
    cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    oneD = tmp_path / "oneD"
    oneD.mkdir(parents=True, exist_ok=True)

    # Build simple signals of length multiple of 1024 so no trimming
    N = 2048  # gives 2 segments per file
    # Each .mat must contain a variable name that includes 'DE'
    # The code uses read_data[var_DE].T then slices along axis=1.
    # Store as (N, 1) so .T -> (1, N)
    sig0 = (np.zeros((N, 1))).astype(np.float32)
    sig1 = (np.arange(N).reshape(N, 1)).astype(np.float32)
    sig2 = (np.sin(2*np.pi*np.arange(N)/64).reshape(N, 1)).astype(np.float32)

    # Filenames must start with the numeric label followed by '_' so split('_')[0] works
    savemat(oneD / "0_DE_sample.mat", {"X_DE_time": sig0})
    savemat(oneD / "1_DE_sample.mat", {"X_DE_time": sig1})
    savemat(oneD / "2_DE_sample.mat", {"X_DE_time": sig2})

    yield oneD

    # Cleanup (pytest tmp dirs are auto-removed, but this restores CWD)
    os.chdir(cwd)


def count_segments_per_file(N=2048, seg=1024):
    return N // seg  # 2 here


# ---------- Tests for low-level utilities ----------

def test_preprocessing_identity():
    x = np.random.randn(4, 1024, 1).astype(np.float32)
    y = mh.preprocessing(x)
    assert y is x  # function returns original reference


def test_oneD_Fourier_shape_and_nonnegativity():
    # Build 2 samples, 1024 each
    x = np.stack([
        np.zeros((1024, 1), dtype=np.float32),
        np.ones((1024, 1), dtype=np.float32),
    ], axis=0)  # (2,1024,1)

    y = mh.oneD_Fourier(x)
    assert y.shape == (2, 1024, 1)
    # Magnitudes should be real and non-negative
    assert np.all(y >= 0)
    # DC component of ones should be > 0
    assert y[1].max() > 0


# ---------- Tests for MatHandler.read_mat() ----------

def test_read_mat_parses_segments_and_labels(tmp_oneD_dir):
    handler = mh.MatHandler(is_oneD_Fourier=False)
    data, labels = handler.read_mat()

    # We created three files: labels 0, 1, 2, each with 2 segments
    expected_segments = 3 * count_segments_per_file()
    assert data.shape == (expected_segments, 1024, 1)
    assert labels.shape == (expected_segments,)

    # Labels should include exactly {0,1,2} and repeat per segment count
    unique = np.unique(labels)
    assert set(unique.tolist()) == {0, 1, 2}

    # Count per label should be equal
    for lab in [0, 1, 2]:
        assert np.sum(labels == lab) == count_segments_per_file()


# ---------- Tests for MatHandler.split_dataset() ----------

def test_split_dataset_shapes_and_consistency(tmp_oneD_dir):
    # No Fourier for this test
    handler = mh.MatHandler(is_oneD_Fourier=False)

    X_tr, y_tr, X_val, y_val, X_te, y_te = handler.split_dataset(False)

    # Basic shape checks
    assert X_tr.ndim == 3 and X_tr.shape[1:] == (1024, 1)
    assert X_val.ndim == 3 and X_val.shape[1:] == (1024, 1)
    assert X_te.ndim == 3 and X_te.shape[1:] == (1024, 1)

    # Total sample count preserved
    total = X_tr.shape[0] + X_val.shape[0] + X_te.shape[0]
    data, _ = handler.read_mat()
    assert total == data.shape[0]

    # Labels align to data counts
    assert y_tr.shape[0] == X_tr.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert y_te.shape[0] == X_te.shape[0]


def test_split_dataset_with_fourier(tmp_oneD_dir):
    handler = mh.MatHandler(is_oneD_Fourier=True)
    X_tr, _, X_val, _, X_te, _ = handler.X_train, handler.y_train, handler.X_val, handler.y_val, handler.X_test, handler.y_test

    # Still 3D after Fourier
    assert X_tr.ndim == 3 and X_tr.shape[1:] == (1024, 1)
    assert X_val.ndim == 3 and X_val.shape[1:] == (1024, 1)
    assert X_te.ndim == 3 and X_te.shape[1:] == (1024, 1)

    # Fourier produces non-negative magnitudes
    assert np.all(X_tr >= 0)
    assert np.all(X_val >= 0)
    assert np.all(X_te >= 0)


# ---------- Tests for get_Data_By_Label() ----------

def test_get_data_by_label_selection_and_shuffle(tmp_oneD_dir):
    handler = mh.MatHandler(is_oneD_Fourier=False)

    # Ask for only label 2 (plus the function always includes label 0)
    data, labels = mh.get_Data_By_Label(handler, pattern='full', label_list=[2])

    # Should contain only {0, 2}
    assert set(np.unique(labels).tolist()) <= {0, 2}
    assert data.shape[0] == labels.shape[0]

    # Deterministic shuffle: calling again should produce same order
    data2, labels2 = mh.get_Data_By_Label(handler, pattern='full', label_list=[2])
    assert np.array_equal(labels, labels2)
    assert np.array_equal(data, data2)


# ---------- Tests for load_Dataset_Original() ----------

def test_load_dataset_original_without_tf(monkeypatch, tmp_oneD_dir):
    """
    Force the fallback path (_HAVE_TF=False) and verify it returns numpy arrays.
    """
    monkeypatch.setattr(mh, "_HAVE_TF", False, raising=True)

    arr = mh.load_Dataset_Original(label_list=[1, 2], batch_size=4, is_oneD_Fourier=False, pattern='train')
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3 and arr.shape[1:] == (1024, 1)


@pytest.mark.skipif(not mh._HAVE_TF, reason="TensorFlow not available in this environment")
def test_load_dataset_original_with_tf(tmp_oneD_dir):
    """
    When TF is present, ensure we get a tf.data.Dataset that batches correctly.
    """
    ds = mh.load_Dataset_Original(label_list=[1, 2], batch_size=3, is_oneD_Fourier=False, pattern='train')

    import tensorflow as tf
    assert isinstance(ds, tf.data.Dataset)

    # Pull one batch and verify shape
    batch = next(iter(ds))
    # preprocessing is identity, dataset yields data only (no labels)
    assert batch.shape[1:] == (1024, 1)
    assert batch.shape[0] <= 3  # last batch may be smaller
