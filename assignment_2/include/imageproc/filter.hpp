#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include<optional>

namespace imageproc {

    class BoxFilter {
    public:
        // constructor kernel size
        explicit BoxFilter(int size);

        cv::Mat apply(const cv::Mat& input) const;

    private:
        int size_;   // filter size (m)
        cv::Mat kernel_; 
    };

    class SharpenFilter {
    public:
        explicit SharpenFilter(double p, int size = 3);

        cv::Mat apply(const cv::Mat& input, std::optional<int> gaussian_blur = {}) const;

    private:
        double p_;      
        BoxFilter box_;   // internal box filter for blurring
    };

}
