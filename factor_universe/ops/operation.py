"""
numba加速后的运算函数
"""
import pandas as pd
import numpy as np
import numba as nb
from itertools import product
import tqdm


# method_in_group
@nb.njit()
def mean(arr):
    return np.mean(arr)


@nb.njit()
def diff(big, small):
    return big - small


@nb.njit()
def roll_1D(a, w, step=1):
    """generate 1-d rolling window from given n-d array

    Args:
        a (N-d array): N-d array to be roll cut
        w (int): Width of rolling window
        step (int, Optional): Step size. Default to 1.

    Returns:
        (n+1)-d array: The first axis contains each generated rolling window, the remain axis are the same as the input N-d array (a). 
    """
    shape = ((a.shape[-1] - w) // step + 1,) + (w,) # (滚动窗口数，窗宽)
    strides = (a.strides[-1] * step,) + a.strides[-1:] # (跨越最后一个维度所需的字节数*步长，跨越最后一个维度所需的字节数)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides) # Eg. 输入a为1-d array时，返回2-darray，每行是移动窗口的元素，按窗口顺序从上到下排列


@nb.extending.overload(np.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """ Implement np.nan_to_num in numba pipeline. """
    if isinstance(x, nb.types.Array):
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if copy:
                out = np.copy(x).reshape(-1)
            else:
                out = x.reshape(-1)
            for i in range(len(out)):
                if np.isnan(out[i]):
                    out[i] = nan
                if posinf is not None and np.isinf(out[i]) and out[i] > 0:
                    out[i] = posinf
                if neginf is not None and np.isinf(out[i]) and out[i] < 0:
                    out[i] = neginf
            return out.reshape(x.shape)

    else:
        def nan_to_num_impl(x, copy=True, nan=0.0, posinf=None, neginf=None):
            if np.isnan(x):
                return nan
            if posinf is not None and np.isinf(x) and x > 0:
                return posinf
            if neginf is not None and np.isinf(x) and x < 0:
                return neginf
            return x

    return nan_to_num_impl


@nb.njit()
def roll_cut_agg(a, w, cutQtlL, cutQtlR, step, in_group):
    """rolling cut and aggregate

    Args:
        a (N-d array): N-d array to be roll cut
        w (int): Width of rolling window
        quantile (float): Cut quantile
        step (int): Step size.
        in_group (function): Aggregational function in each cutting group
        dirct (str): ['top', 'bot]

    Returns:
        1-d array: Value of local factors. Length equals to numbers of rolling windows.
    """
    
    res = np.empty((a.shape[-1] - w) // step + 1) # 创建结果向量，长度为滚动窗口的数量
    # m = int(w * cutQtl // step)  # cut size
    
    # for i, arr in enumerate(roll_1D(a, w, step)):
        
    #     nan_to_num(arr[::2], copy=False, nan=np.nanmedian(arr[::2])) # 用中位数替换tool的缺失值
    #     nan_to_num(arr[1::2], copy=False, nan=np.nanmedian(arr[1::2])) # 用中位数替换obj的缺失值
        
    #     idx = arr[::2].argsort()  # 根据tool从小到大排序的index向量
    #     res[i] = in_group(arr[1::2][idx[:m]]) if dirct == 'bot' else in_group(arr[1::2][idx[-m:]]) # 对每个移动窗口，根据dirct参数判断截取窗口头部还是尾部，并返回截取片段传入聚合函数后的返回值
    
    cutPointL, cutPointR = int( cutQtlL * w // step ), int( w * cutQtlR // step )
    for i, arr in enumerate(roll_1D(a, w, step)):
    
        # nan_to_num(arr[::2], copy=False, nan=np.nanmedian(arr[::2])) # 用中位数替换tool的缺失值
        # nan_to_num(arr[1::2], copy=False, nan=np.nanmedian(arr[1::2])) # 用中位数替换obj的缺失值
        arr[::2][np.isnan(arr[::2])] = np.nanmedian(arr[::2]) # 用中位数替换tool的缺失值
        arr[1::2][np.isnan(arr[1::2])] = np.nanmedian(arr[1::2]) # 用中位数替换obj的缺失值
            
        idx = arr[::2].argsort()  # 根据tool从小到大排序的index向量
        res[i] = in_group(arr[1::2][idx[cutPointL:cutPointR]])      
    return res

