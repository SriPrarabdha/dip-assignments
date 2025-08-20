from src.cce import ConnectedComponentExtractor

import matplotlib.pyplot as plt
from typing import Optional
import os

def run_task_4(image_path:str, save_dir:Optional[bool]=None):

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    extractor = ConnectedComponentExtractor(image_path)
    extractor.plot_results(save_dir)