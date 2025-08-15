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

# img = Image.open("images/coins.png")
# img_arr = np.array(img)

# hist = np.zeros(256, dtype='uint8')

# for i in range(len(img_arr)):
#     for j in range(len(img_arr[0])):
#         hist[img_arr[i][j]]+=1

@dataclass
class brute_force_soln:
    def mean(self, cls: int , thres:int):
        pro = 0
        total_pixels = 0
        if(cls):
            for i in range(thres+1 , 256):
                total_pixels+=hist[i]
                pro+=(i*hist[i])
        else:
            for i in range(0, thres+1):
                total_pixels+=hist[i]
                pro+=(i*hist[i])


        return pro/total_pixels

    def var(self, cls:int , thres:int, mean:float):
        pro = 0
        total_pixels = 0
        if(cls):
            for i in range(thres+1 , 256):
                total_pixels+=hist[i]
                pro+=((i-mean)*hist[i])
        else:
            for i in range(0, thres+1):
                total_pixels+=hist[i]
                pro+=((i-mean)*hist[i])


        return pro/total_pixels , total_pixels

    def interclass_var(self, thres:int):
        mean_1 = self.mean(0, thres)
        mean_2 = self.mean(1 , thres)

        var_1, n1 = self.var(0, thres , mean_1)
        var_2 , n2= self.var(1, thres , mean_2)

        n = len(hist) * len(hist[0])

        return (var_1*n1 + var_2*n2)/n

    def main(self):
        least_var = 100
        t_opt = 256

        for t in range(0, 256):
            it_var = self.interclass_var(t)
            if(it_var<least_var):
                least_var = it_var
                t_opt = t


@dataclass
class better_soln:
    pixels = np.arange(256, dtype='uint8')

    def mean(self, cls: int , thres:int):
        if(cls): mask = self.pixels>thres
        else : mask = self.pixels<=thres

        total_pixels = np.sum(hist[mask])
        pro = (self.pixels[mask] * hist[mask]).sum()

        return pro/total_pixels

    def var(self, cls:int , thres:int, mean:float):
        if(cls): mask = self.pixels>thres
        else : mask = self.pixels<=thres

        total_pixels = np.sum(hist[mask])
        pro = (np.power(self.pixels[mask] - mean) * hist[mask]).sum()

        return pro/total_pixels , total_pixels

    def interclass_var(self, thres:int):
        mean_1 = self.mean(0, thres)
        mean_2 = self.mean(1 , thres)

        var_1, n1 = self.var(0, thres , mean_1)
        var_2 , n2= self.var(1, thres , mean_2)

        n = len(hist) * len(hist[0])

        return (var_1*n1 + var_2*n2)/n

    def main(self):
        least_var = 100
        t_opt = 256

        for t in range(0, 256):
            it_var = self.interclass_var(t)
            if(it_var<least_var):
                least_var = it_var
                t_opt = t


from dataclasses import dataclass
import numpy as np

from numba import njit

@njit
def get_threshold_numba(cum_count, cum_sum, cum_sum_sq):
    least_var = 1e18
    t_opt = -1
    total_pixels = cum_count[-1]

    for t in range(256):
        # compute mean0
        n0 = cum_count[t]
        sum0 = cum_sum[t]
        mean0 = sum0 / n0 if n0 > 0 else 0.0

        # compute mean1
        n1 = total_pixels - n0
        sum1 = cum_sum[-1] - sum0
        mean1 = sum1 / n1 if n1 > 0 else 0.0

        # compute variances
        sum0_sq = cum_sum_sq[t]
        var0 = (sum0_sq - 2 * mean0 * sum0 + mean0**2 * n0) / n0 if n0 > 0 else 0.0

        sum1_sq = cum_sum_sq[-1] - sum0_sq
        var1 = (sum1_sq - 2 * mean1 * sum1 + mean1**2 * n1) / n1 if n1 > 0 else 0.0

        # within-class variance
        wc_var = (var0 * n0 + var1 * n1) / total_pixels
        if wc_var < least_var:
            least_var = wc_var
            t_opt = t

    print("inter class ousu threshold = ", t_opt)
    return t_opt


class optimal_otsu_ic:

    def __init__(self, image_path:str , offset:Optional[int]=None):
        
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

    def mean(self, cls: int, thres: int):
        if cls:
            total_pixels = self.cum_count[-1] - self.cum_count[thres]
            pro = self.cum_sum[-1] - self.cum_sum[thres]

        else:
            total_pixels = self.cum_count[thres]
            pro = self.cum_sum[thres]

        return pro / total_pixels if total_pixels else 0

    def var(self, cls: int, thres: int, mean: float):
        if cls:
            total_pixels = self.cum_count[-1] - self.cum_count[thres]
            sum_i = self.cum_sum[-1] - self.cum_sum[thres]
            sum_i2 = self.cum_sum_sq[-1] - self.cum_sum_sq[thres]

        else:
            total_pixels = self.cum_count[thres]
            sum_i = self.cum_sum[thres]
            sum_i2 = self.cum_sum_sq[thres]

        if total_pixels == 0:
            return 0, 0
        
        var = (sum_i2 - 2 * mean * sum_i + mean**2 * total_pixels) / total_pixels
        return var, total_pixels

    def interclass_var(self, thres: int):
        mean_1 = self.mean(0, thres)
        mean_2 = self.mean(1, thres)
        var_1, n1 = self.var(0, thres, mean_1)
        var_2, n2 = self.var(1, thres, mean_2)
        
        return (var_1 * n1 + var_2 * n2) / self.cum_count[-1]

    def get_threshold(self):
        t_opt = get_threshold_numba(self.cum_count, self.cum_sum, self.cum_sum_sq)
        # print("my Optimal threshold:", t_opt)
        return t_opt
    
    def plot_image(self):
        t_opt = self.get_threshold()
        plt.imshow(self.img_arr > t_opt, cmap='gray')
        plt.axis('off')
        plt.show()


def base_case(img_path:str):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    t = time()
    for _ in range(10000):
        optimal_thresh, thresh_img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    print("average per call:", (time() - t) / 10000)


if __name__ == "__main__":

    a = optimal_otsu_ic("images/coins.png", 0)
    t = time()
    # a.get_threshold()
    # for _ in range(10000):
    #     thres = a.get_threshold()
    a.plot_image()
    # print("average per call:", (time() - t) / 10000)

    # plt.plot()

    # base_case()
