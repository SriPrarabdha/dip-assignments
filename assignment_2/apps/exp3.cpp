#include "apps/exp3.hpp"
#include <imageproc/filter.hpp>
#include <imageproc/io.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace apps {

void sharpen_experiment(const std::string& image_path,std::optional<std::string> save_dir ) {
    auto img = imageproc::io::loadImage(image_path);
    imageproc::io::showImage("original image", img);

    imageproc::BoxFilter bf(29);
    auto blurred_img = bf.apply(img);

    imageproc::io::showImage("original image", blurred_img);
    imageproc::SharpenFilter sf(10, 229);

    auto sharp_img = sf.apply(blurred_img, 17);
    imageproc::io::showImage("sharp image", img);

    
    }
}

