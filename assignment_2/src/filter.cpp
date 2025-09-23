#include "imageproc/filter.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <optional>
namespace imageproc {

    BoxFilter::BoxFilter(int size) : size_(size) {
        if (size <= 0 || size % 2 == 0) {
            throw std::invalid_argument("BoxFilter size must be a positive odd integer.");
        }
    
        kernel_ = cv::Mat::ones(size_, size_, CV_32F) / static_cast<float>(size_ * size_);
    }

    cv::Mat BoxFilter::apply(const cv::Mat& input) const {
        cv::Mat output;
        cv::filter2D(input, output, -1, kernel_, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
        return output;
    }

    SharpenFilter::SharpenFilter(double p, int size) : p_(p), box_(size) {
        if (p < 0.0) {
            throw std::invalid_argument("Sharpen parameter p must be in range [0,1].");
        }
    }

    cv::Mat SharpenFilter::apply(const cv::Mat& input, std::optional<int> gaussian_blur) const {
        // Blur with boxfilter
        cv::Mat blurred;
        if (gaussian_blur.has_value()) {
            cv::GaussianBlur(input, blurred, cv::Size(gaussian_blur.value(), gaussian_blur.value()), 0);
        } else {
            blurred = box_.apply(input);
        }


        // masking
        cv::Mat output;
        cv::addWeighted(input, 1.0 + p_, blurred, -p_, 0, output);

        return output;
    }

}
