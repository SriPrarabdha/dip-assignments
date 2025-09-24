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
    imageproc::SharpenFilter sf(0.8, 5);

    auto sharp_img_box = sf.apply(img);
    auto sharp_img_gaussian = sf.apply(img, 17);
    std::cout<<"fuck";
    if(save_dir.has_value()){
        std::string s = *save_dir;
        // imageproc::io::saveImage(s+"/exp_3_blurred.png" , blurred_img);
        imageproc::io::saveImage(s+"/exp_3_sharp_box.png" , sharp_img_box);
        imageproc::io::saveImage(s+"/exp_3_sharp_gaussian.png" , sharp_img_gaussian);
    }
    imageproc::io::showImage("sharp image box", sharp_img_box);
    imageproc::io::showImage("sharp image gaussian", sharp_img_gaussian);

    
    }
}
