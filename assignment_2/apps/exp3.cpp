#include "apps/exp3.hpp"
#include <imageproc/filter.hpp>
#include <imageproc/io.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace apps {

void sharpen_experiment(const std::string& image_path,std::optional<std::string> save_dir ) {
    auto img = imageproc::io::loadImage(image_path);
    imageproc::io::showImage("original image", img);

    // imageproc::BoxFilter bf(19);
    // auto blurred_img = bf.apply(img);

    // imageproc::io::showImage("original image", blurred_img);
    imageproc::SharpenFilter sf(0.8, 29);

    // auto sharp_img = sf.apply(blurred_img, 17);
    auto sharp_img = sf.apply(img, 17);
    if(save_dir.has_value()){
        std::string s = *save_dir;
        // imageproc::io::saveImage(s+"/exp_3_blurred.png" , blurred_img);
        imageproc::io::saveImage(s+"/exp_3_sharp.png" , sharp_img);
    }
    imageproc::io::showImage("sharp image", sharp_img);

    
    }
}

