from PIL import Image
import numpy as np
from typing import Optional, Tuple
import os
from tqdm import tqdm
from time import time
from numba import njit, prange
import matplotlib.pyplot as plt
import cv2
# from concurrent.futures import ThreadPoolExecutor
# import multiprocessing as mp

@njit
def compute_histogram_numba(arr):
    hist = np.zeros(256, dtype=np.uint32)
    flat_arr = arr.flatten()
    for i in range(flat_arr.size):
        hist[flat_arr[i]] += 1
    return hist

@njit
def otsu_threshold_numba(hist):
    """Numba-optimized Otsu threshold computation"""
    pixels = np.arange(256, dtype=np.float64)
    cum_count = np.cumsum(hist)
    cum_sum = np.cumsum(hist * pixels)
    
    total_pixels = cum_count[-1]
    if total_pixels == 0:
        return 0
    
    max_var = -1.0
    t_opt = 0
    for t in range(256):
        w1 = cum_count[t] / total_pixels
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            continue
        
        mean1 = cum_sum[t] / cum_count[t]
        mean2 = (cum_sum[-1] - cum_sum[t]) / (total_pixels - cum_count[t])
        var_between = w1 * w2 * (mean1 - mean2) ** 2
        
        if var_between > max_var:
            max_var = var_between
            t_opt = t
    return t_opt

@njit(parallel=True)
def process_windows_integral_parallel(img_arr, integral_hist, window_size, stride, window_positions):

    output_img = np.zeros_like(img_arr)
    pad = window_size // 2
    H, W = img_arr.shape
    pixels = np.arange(256, dtype=np.float64)
    
    for idx in prange(len(window_positions)):
        i, j = window_positions[idx]
        
        x1 = max(0, i - pad)
        x2 = min(H - 1, i + pad)
        y1 = max(0, j - pad)
        y2 = min(W - 1, j + pad)
        
        hist = (integral_hist[x2+1, y2+1] - integral_hist[x1, y2+1] - 
                integral_hist[x2+1, y1] + integral_hist[x1, y1])
        
        t_local = otsu_threshold_numba(hist)
        
        end_i = min(i + stride, H)
        end_j = min(j + stride, W)
        
        for ii in range(i, end_i):
            for jj in range(j, end_j):
                output_img[ii, jj] = 255 if img_arr[ii, jj] > t_local else 0
    
    return output_img

class fast_otsu_adapt:
    
    def __init__(self, image_path: str, offset: Optional[int] = None):
        self.img_arr = np.array(Image.open(image_path), dtype=np.uint8)
        
        if offset is not None:
            self.img_arr = self.img_arr.astype(np.int32) + offset
            self.img_arr = np.clip(self.img_arr, 0, 255).astype(np.uint8)
        
        self.H, self.W = self.img_arr.shape
        self._build_integral_histogram()
    
    def _build_integral_histogram(self):
        self.integral_hist = np.zeros((self.H+1, self.W+1, 256), dtype=np.uint32)
        
        for i in range(self.H):
            row_hist = np.zeros(256, dtype=np.uint32)
            for j in range(self.W):
                row_hist[self.img_arr[i, j]] += 1
                self.integral_hist[i+1, j+1] = (
                    self.integral_hist[i, j+1] + row_hist
                )
    
    def adaptive_otsu_vectorized(self, window_size=15):
        """Vectorized implementation"""
        stride = int(window_size * 0.8)
        pad = window_size // 2
        
        i_positions = np.arange(0, self.H, stride)
        j_positions = np.arange(0, self.W, stride)
        window_positions = np.array([(i, j) for i in i_positions for j in j_positions])
        
        #integral histograms
        output_img = process_windows_integral_parallel(
            self.img_arr, self.integral_hist, window_size, stride, window_positions
        )
        
        return output_img
    
    
    def plot_image(self, window_size=15, save_dir: Optional[str] = None):  
        start_time = time()
        adaptive_img = self.adaptive_otsu_vectorized(window_size)
        end_time = time()
        
        print(f"Processing time: {end_time - start_time:.3f} seconds")
        
        plt.imshow(adaptive_img, cmap='gray')
        plt.axis('off')
        plt.title(f'Adaptive Otsu - Window Size: {window_size}')
        
        if save_dir:
            plt.savefig(f"{save_dir}/task3_adapt_otsu_binary_{window_size}.png", 
                       bbox_inches='tight', dpi=150)
        
        plt.show()
        plt.close()