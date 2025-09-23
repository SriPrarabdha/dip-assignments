#include "imageproc/filter.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

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

}
