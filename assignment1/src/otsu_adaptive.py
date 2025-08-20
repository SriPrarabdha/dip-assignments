from PIL import Image
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import os
from tqdm import tqdm
from time import time
from numba import njit
import matplotlib.pyplot as plt
import cv2

class good_enough_otsu_adapt:
    def __init__(self, image_path: str, offset: Optional[int] = None):
        self.img_arr = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)  # force grayscale

        if offset is not None:
            self.img_arr = self.img_arr.astype(np.int32) + offset
            self.img_arr = np.clip(self.img_arr, 0, 255).astype(np.uint8)

    def compute_histogram(self, arr):
        hist = np.zeros(256, dtype=np.uint32)
        for val in arr.flatten():
            hist[val] += 1
        return hist

    def otsu_threshold_from_hist(self, hist):
        pixels = np.arange(256, dtype=np.float64)
        cum_count = np.cumsum(hist, dtype=np.uint32)
        cum_sum = np.cumsum(hist * pixels, dtype=np.float64)

        total_pixels = cum_count[-1]
        if total_pixels == 0:
            return 0

        global_mean = cum_sum[-1] / total_pixels

        max_var = -1
        t_opt = 0
        for t in range(256):
            w1 = cum_count[t] / total_pixels
            w2 = 1 - w1
            if w1 == 0 or w2 == 0:
                continue

            mean1 = cum_sum[t] / cum_count[t]
            mean2 = (cum_sum[-1] - cum_sum[t]) / (total_pixels - cum_count[t])
            var_between = w1 * w2 * (mean1 - mean2) ** 2

            if var_between > max_var:
                max_var = var_between
                t_opt = t

        return t_opt

    def adaptive_otsu(self, window_size=15):
        pad = window_size // 2
        padded_img = np.pad(self.img_arr, pad, mode="reflect")
        output_img = np.zeros_like(self.img_arr)

        for i in range(self.img_arr.shape[0]):
            for j in range(self.img_arr.shape[1]):
                window = padded_img[i:i + window_size, j:j + window_size]
                hist = self.compute_histogram(window)
                t_local = self.otsu_threshold_from_hist(hist)
                output_img[i, j] = 255 if self.img_arr[i, j] > t_local else 0

        return output_img

    def plot_image(self, window_size=15, save_dir:Optional[str] = None):
        adaptive_img = self.adaptive_otsu(window_size)
        plt.imshow(adaptive_img, cmap='gray')
        plt.axis('off')
        if(save_dir): plt.savefig(f"{save_dir}/task3_adapt_otsu_binary_{window_size}.png")
        
        plt.show()
        plt.close()




class optimal_otsu_adapt:
    def __init__(self, image_path: str, offset: Optional[int] = None):
        self.img_arr = np.array(Image.open(image_path), dtype=np.uint8)

        if offset is not None:
            self.img_arr = self.img_arr.astype(np.int32) + offset
            self.img_arr = np.clip(self.img_arr, 0, 255).astype(np.uint8)

        self.H, self.W = self.img_arr.shape
        self.pixels = np.arange(256, dtype=np.float64) 

        # Build integral histogram (H+1, W+1, 256)
        self.integral_hist = np.zeros((self.H+1, self.W+1, 256), dtype=np.uint32)
        for i in range(self.H):
            row_hist = np.zeros(256, dtype=np.uint32)
            for j in range(self.W):
                row_hist[self.img_arr[i, j]] += 1
                self.integral_hist[i+1, j+1] = (
                    self.integral_hist[i, j+1] +
                    row_hist
                )

    def get_histogram_from_window(self, x1, y1, x2, y2):

        return (
            self.integral_hist[x2+1, y2+1]
            - self.integral_hist[x1, y2+1]
            - self.integral_hist[x2+1, y1]
            + self.integral_hist[x1, y1]
        )

    def otsu_threshold_from_hist(self, hist):
        cum_count = np.cumsum(hist)
        cum_sum = np.cumsum(hist * self.pixels)

        total_pixels = cum_count[-1]
        if total_pixels == 0:
            return 0

        max_var = -1
        t_opt = 0
        for t in range(256):
            w1 = cum_count[t] / total_pixels
            w2 = 1 - w1
            if w1 == 0 or w2 == 0:
                continue
            mean1 = cum_sum[t] / cum_count[t]
            mean2 = (cum_sum[-1] - cum_sum[t]) / (total_pixels - cum_count[t])
            var_between = w1 * w2 * (mean1 - mean2) ** 2
            if var_between > max_var:
                max_var = var_between
                t_opt = t
        return t_opt

    def adaptive_otsu(self, window_size=15):
        pad = window_size // 2
        output_img = np.zeros_like(self.img_arr)

        for i in range(self.H):
            x1 = max(0, i - pad)
            x2 = min(self.H - 1, i + pad)
            for j in range(self.W):
                y1 = max(0, j - pad)
                y2 = min(self.W - 1, j + pad)
                hist = self.get_histogram_from_window(x1, y1, x2, y2)
                t_local = self.otsu_threshold_from_hist(hist)
                output_img[i, j] = 255 if self.img_arr[i, j] > t_local else 0

        return output_img

    def plot_image(self, window_size=15, save_dir:Optional[str] = None):
        adaptive_img = self.adaptive_otsu(window_size)
        plt.imshow(adaptive_img, cmap='gray')
        plt.axis('off')
        
        if(save_dir): plt.savefig(f"{save_dir}/task3_adapt_otsu_binary_{window_size}.png")
        
        plt.show()
        plt.close()
