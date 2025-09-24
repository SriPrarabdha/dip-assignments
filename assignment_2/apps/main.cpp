#include <iostream>

#include "apps/exp1.hpp"
#include "apps/exp2.hpp"
#include "apps/exp3.hpp"

using namespace std;

int main() {

    //exp 1
    string img_path = "../ip_images/moon_noisy.png";
    string op_dir = "../op_images";

    // apps::run_blur_experiment(img_path, op_dir);

    //exp 2
    img_path = "../ip_images/flowers.png";
    // apps::scale_rotate_experiment(img_path, op_dir);

    img_path = "../ip_images/study.png";
    apps::sharpen_experiment(img_path, op_dir);

    return 0;
}
