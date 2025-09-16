#pragma once
#include <vector>
#include <opencv2/core.hpp>

namespace imageproc {

    class BoxFilter {
    public:
        // constructor with kernel size (m x m)
        explicit BoxFilter(int size);

        // apply filter to an image
        cv::Mat apply(const cv::Mat& input) const;

    private:
        int size_;   // filter size (m)
        cv::Mat kernel_;  // box filter kernel
    };

}
