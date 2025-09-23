#include "apps/exp3.hpp"
#include <imageproc/filter.hpp>
#include <imageproc/io.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace apps {

void sharpen_experiment(const std::string& image_path) {
    auto img = imageproc::io::loadImage(image_path);
    imageproc::io::showImage("original image", img);
    imageproc::SharpenFilter sf(10, 229);

    auto sharp_img = sf.apply(img, 17);
    imageproc::io::showImage("sharp image", img);

    
    }
}

