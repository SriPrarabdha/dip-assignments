from tasks.task1 import plot_hist
from tasks.task2 import run_task_2
from tasks.task3 import run_task_3
from tasks.task4 import run_task_4

def main():

    # task 1
    img_path = "ip_images/coins.png"
    plot_hist(img_path, save_dir="op_images")

    # task 2
    img_path = "ip_images/coins.png"
    run_task_2(input_path=img_path, save_dir="op_images" , profiling=True, debug = True)    

    #task 3
    img_path = "ip_images/sudoku.png"
    run_task_3(input_path=img_path, save_dir="op_images")

    #task 4
    img_path = "ip_images/quote.png"
    run_task_4(image_path=img_path , save_dir="op_images")

if __name__ == "__main__":
    main()




