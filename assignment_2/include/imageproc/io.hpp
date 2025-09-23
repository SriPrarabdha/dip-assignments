#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <optional>

namespace imageproc {

    namespace io {
        // load an image (grayscale)
        cv::Mat loadImage(const std::string& path, std::optional<int> greyscale = {});

        // save image to file
        void saveImage(const std::string& path, const cv::Mat& img);

        // optional: show image in a window
        void showImage(const std::string& title, const cv::Mat& img);
    }

}
