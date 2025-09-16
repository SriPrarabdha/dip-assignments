#include "imageproc/filter.hpp"
#include <opencv2/imgproc.hpp>

namespace imageproc {

    BoxFilter::BoxFilter(int size) : size_(size) {
        // TODO: initialize kernel_ as an m x m matrix with all values = 1/(m*m)
    }

    cv::Mat BoxFilter::apply(const cv::Mat& input) const {
        // TODO: use cv::filter2D with kernel_ on input
        return input.clone(); // placeholder
    }

}
