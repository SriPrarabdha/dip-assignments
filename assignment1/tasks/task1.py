from PIL import Image
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from typing import Optional


def plot_hist(img_path:str, save_dir:Optional[str]=None):
    img = Image.open(img_path)

    img_arr = np.array(img)

    print("image array shape = " ,img_arr.shape)

    hist = [0]*256

    for i in range(len(img_arr)):
        for j in range(len(img_arr[0])):
            hist[img_arr[i][j]]+=1

    # pprint(hist)

    plt.bar(range(256), hist, width=1.0, color='gray')
    plt.xlabel("Intensity Level")
    plt.ylabel("Frequency")
    plt.title("Pixel Histogram")
    
    if(save_dir): plt.savefig(f"{save_dir}/task1_histogram.png")
    
    plt.show()

    total_pixels = len(img_arr) * len(img_arr[0])
    avg_from_hist = sum(i * hist[i] for i in range(256)) / total_pixels

    avg_direct = np.sum(img_arr) / total_pixels

    print(f"Average intensity from histogram: {avg_from_hist:.2f}")
    print(f"Average intensity direct: {avg_direct:.2f}")