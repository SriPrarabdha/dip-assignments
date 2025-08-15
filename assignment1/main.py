from tasks.task1 import plot_hist

from src.otsu_bc import optimal_otsu_bc
from src.otsu_ic import optimal_otsu_ic
from src.otsu_adaptive import optimal_otsu_adapt

from time import time

def main():

    # task 1
    img_path = "images/coins.png"
    plot_hist(img_path)

    # task 2.1

    img_path = "images/coins.png"
    otsu_inter_class = optimal_otsu_ic(img_path, 0)
    t = time()

    for _ in range(10000):
        thres = otsu_inter_class.get_threshold()
    print("average per call:", (time() - t) / 10000)
    otsu_inter_class.plot_image()

    #task 2.2

    otsu_btw_class = optimal_otsu_bc(img_path , 20)
    t = time()

    for _ in range(10000):
        thres = otsu_btw_class.get_threshold()
    print("average per call:", (time() - t) / 10000)
    otsu_btw_class.plot_image()

    # base_case()

    #task 3
    img_path = "images/coins.png"
    otsu_adapt = optimal_otsu_adapt(img_path)

    otsu_adapt.plot_image(window_size=5)






if __name__ == "__main__":
    main()




