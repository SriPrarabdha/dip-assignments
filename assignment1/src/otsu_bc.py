from PIL import Image
import numpy as np
from typing import Optional
from dataclasses import dataclass
import os
from tqdm import tqdm
from time import time
from numba import njit
import matplotlib.pyplot as plt
import cv2


class optimal_otsu_bc:

    def __init__(self, image_path:str , offset:Optional[int]):

        self.img_arr = np.array(Image.open(image_path))

        if offset is not None:
            self.img_arr = self.img_arr.astype(np.int32) + offset
            self.img_arr = np.clip(self.img_arr, 0, 255).astype(np.uint8)

        self.hist = np.zeros(256, dtype=np.uint32)

        for i in range(len(self.img_arr)):
            for j in range(len(self.img_arr[0])):
                self.hist[self.img_arr[i][j]]+=1

        self.pixels = np.arange(256, dtype=np.uint16)
        self.cum_count = np.cumsum(self.hist, dtype=np.uint32)
        self.cum_sum = np.cumsum(self.hist * self.pixels, dtype=np.uint64)
        self.cum_sum_sq = np.cumsum(self.hist * (self.pixels ** 2), dtype=np.uint64)

        self.global_mean = self.cum_sum[-1]/self.cum_count[-1]
        self.global_var = (self.cum_sum_sq[-1] - 2 * self.global_mean * self.cum_sum[-1] + self.global_mean**2 * self.cum_count[-1]) / self.cum_count[-1]

    def mean(self, cls: int, thres: int):
        if cls:
            total_pixels = self.cum_count[-1] - self.cum_count[thres]
            pro = self.cum_sum[-1] - self.cum_sum[thres]

        else:
            total_pixels = self.cum_count[thres]
            pro = self.cum_sum[thres]

        if total_pixels == 0:
            return 0, 0

        return pro / total_pixels , total_pixels
    
    def btwclass_var(self, thres: int):
        mean1, n1 = self.mean(0, thres)
        mean2, n2 = self.mean(1, thres)

        w1 = n1 / self.cum_count[-1]
        w2 = n2 / self.cum_count[-1]

        return w1 * w2 * (mean1 - mean2) ** 2

    def get_threshold(self, debug:Optional[bool] = None):
        max_var = -1
        t_opt = -1
        for t in range(256):
            var = self.btwclass_var(t)
            if var > max_var:
                max_var = var
                t_opt = t
        if(debug): print("Optimal threshold acheived by maximizing btwclass variance:", t_opt)
        
        return t_opt
    
    def plot_image(self, save_dir:str=None, debug:Optional[bool]=None):
        t_opt = self.get_threshold(debug)
        plt.imshow(self.img_arr > t_opt, cmap='gray')
        plt.axis('off')
        if(save_dir): plt.savefig(f"{save_dir}/task2_bc_otsu_binary.png")
        
        plt.show()
        plt.close()
