import numpy as np
import pandas as pd
import scipy
from typing import Union, Literal

def splitter(array, win_len, hop_len, return_df = True):
    N = array.shape[0]
    m = 0
    ids = []
    while m + win_len <= N:
        ids.append([m, m + win_len])
        m = m + hop_len
        
    if return_df:
        return pd.DataFrame([array[i[0]: i[1]] for i in ids])
    else:
        return np.array([array[i[0]: i[1]] for i in ids])
    
def fft_freq_axis(time_len, sampling_freq):
    return scipy.fft.fftfreq(time_len, 1/float(sampling_freq))[0 : time_len // 2]

def zoomed_fft_freq_axis(f_min, f_max, desired_len):
    return np.linspace(f_min, f_max, desired_len)

def z_score_scaler(signals: Union[pd.DataFrame, np.ndarray], axis: Literal[0, 1] = 1, return_df: bool = True, eps: float = 1e-10):
    
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (column-wise) or 1 (row-wise)")

    if isinstance(signals, pd.DataFrame):
        mean = signals.mean(axis=axis)
        std = signals.std(axis=axis).replace(0, eps)
        scaled = signals.sub(mean, axis=1-axis).div(std, axis=1-axis)
        return scaled if return_df else scaled.to_numpy()

    elif isinstance(signals, np.ndarray):
        mean = signals.mean(axis=axis, keepdims=True)
        std = signals.std(axis=axis, keepdims=True)
        std = np.where(std == 0, eps, std)
        return (signals - mean) / std

    else:
        raise TypeError("signals must be a pandas DataFrame or a numpy ndarray")