from ..src.cce import ConnectedComponents

import matplotlib.pyplot as plt
from typing import Optional
import os

def run_task_4(input_path:str, save_dir:Optional[bool]=None):

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cc = ConnectedComponents(input_path)
    labels, sizes = cc.extract_components()
    largest_mask = cc.largest_component_mask()

    print("Component sizes:", sizes)

    plt.plot(); plt.imshow(largest_mask, cmap="gray"); plt.title("Largest")
    plt.show()

    if(save_dir) : plt.savefig(f"{save_dir}/task4_cce.png")

    plt.close()