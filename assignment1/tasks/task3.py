from src.otsu_adaptive import optimal_otsu_adapt, good_enough_otsu_adapt
from src.otsu_adaptive_fast import fast_otsu_adapt
from src.otsu_ic import optimal_otsu_ic

from typing import Optional
import os

def run_task_3(input_path:str, save_dir:Optional[str]=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    otsu_adapt = fast_otsu_adapt(input_path)
    otsu_inter_class = optimal_otsu_ic(input_path)

    #task 3.1

    otsu_adapt.plot_image(window_size=5 , save_dir= save_dir)

    # task 3.2

    otsu_adapt.plot_image(window_size=10 , save_dir= save_dir)

    # task 3.3

    otsu_adapt.plot_image(window_size=25 , save_dir= save_dir)

    # task 3.4

    otsu_adapt.plot_image(window_size=50 , save_dir= save_dir)

    # task 3.5

    otsu_inter_class.plot_image(save_dir=save_dir, task=3)

    

if __name__ == "__main__":
    run_task_3("ip_images/sudoku.png")