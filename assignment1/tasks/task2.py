from src.otsu_bc import optimal_otsu_bc
from src.otsu_ic import optimal_otsu_ic, base_case

from typing import Optional
import time
import os

def run_task_2(input_path:str, save_dir:Optional[str] = None , profiling:Optional[bool] = None, debug:Optional[bool]=True):

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #task 2.1
    otsu_inter_class = optimal_otsu_ic(input_path, 0)
    otsu_inter_class.plot_image(save_dir=save_dir, debug=debug)

    if(profiling):
        t1 = time.time()

        for _ in range(10000):
            thres = otsu_inter_class.get_threshold()
        print("average time taken for minimizing interclass var otsu implementation:", (time.time() - t1) / 10000)

        base_case(img_path=input_path)

    #task 2.2

    otsu_btw_class = optimal_otsu_bc(input_path , 20)
    # t = time()

    # for _ in range(10000):
    #     thres = otsu_btw_class.get_threshold()
    # print("average per call:", (time() - t) / 10000)
    otsu_btw_class.plot_image(save_dir=save_dir, debug=debug)


if __name__ == "__main__":
    run_task_2()






