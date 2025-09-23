#include "apps/exp2.hpp"
#include <imageproc/transforms.hpp>
#include <imageproc/io.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace apps {

void scale_rotate_experiment(const std::string& image_path) {

    auto img = imageproc::io::loadImage(image_path);
    std::cout<<img.channels();
    imageproc::io::showImage("original image",img);

    imageproc::transforms transform;
    int a = 45;
    cv::Mat rotate_img = transform.rotate(img, a);
    cv::Mat scaled_image = transform.upsample(img, 2);

    cv::Mat result1 = transform.rotate(scaled_image, -a);
    cv::Mat result2 = transform.upsample(rotate_img, 2);

    imageproc::io::showImage("result 1 image", result1);
    imageproc::io::showImage("result 2 image", result2);

    cv::Mat r1, r2;
    result1.convertTo(r1, CV_32F);
    result2.convertTo(r2, CV_32F);

    cv::Mat diff;
    cv::absdiff(r1, r2, diff);

    cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);

    cv::Mat diff8u;
    diff.convertTo(diff8u, CV_8U);

    imageproc::io::showImage("diff" , diff8u);
    }
}

