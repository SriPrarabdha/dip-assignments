#pragma once
#include <opencv2/core.hpp>
#include <optional>

namespace imageproc {

    class transforms {
    public:

        cv::Mat rotate(const cv::Mat& input, int angle);

        cv::Mat upsample(const cv::Mat& input, int factor);

        cv::Mat interpolate_bilinear(const cv::Mat& input, int factor);
    };

}
