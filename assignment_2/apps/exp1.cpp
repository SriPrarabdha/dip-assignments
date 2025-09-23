#include "apps/exp1.hpp"
#include <imageproc/filter.hpp>
#include <imageproc/binarize.hpp>
#include <imageproc/io.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>

namespace apps {

void run_blur_experiment(const std::string& image_path, std::optional<std::string> save_dir ) {

    auto img = imageproc::io::loadImage(image_path, 1);

    imageproc::OtsuBinarizer binarizer;
    double opt_var = 0.12;
    auto res = binarizer.apply(img);        
    imageproc::io::showImage("thresholded image", res.first);

    for (int m : {5, 29, 129}) {
        imageproc::BoxFilter bf(m);
        auto blurred_img = bf.apply(img);

        auto binary_res = binarizer.apply(blurred_img);

        std::cout << "Filter size " << m << " → optimal variance = " << binary_res.second << "\n";

        // imageproc::io::saveImage("out_blur_" + std::to_string(m) + ".png", blurred);
        // imageproc::io::saveImage("out_bin_" + std::to_string(m) + ".png", binary);

        imageproc::io::showImage("thresholded image", binary_res.first);
    }
}

} // namespace apps
