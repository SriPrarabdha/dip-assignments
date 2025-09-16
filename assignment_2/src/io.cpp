#include "imageproc/io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace imageproc::io {

    cv::Mat loadImage(const std::string& path) {
        // TODO: load as grayscale
        return cv::imread(path, cv::IMREAD_GRAYSCALE);
    }

    void saveImage(const std::string& path, const cv::Mat& img) {
        cv::imwrite(path, img);
    }

    void showImage(const std::string& title, const cv::Mat& img) {
        cv::imshow(title, img);
        cv::waitKey(0);
    }

}
