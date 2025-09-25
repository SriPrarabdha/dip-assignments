#pragma once
#include <opencv2/core.hpp>
#include <optional>

namespace imageproc {

    class OtsuBinarizer {
    public:
        // apply Otsu's thresholding on input image
        std::pair<cv::Mat,std::pair<int, int>> apply(const cv::Mat& input, std::optional<int> plot={}, std::optional<int> profile = {}) const;

    private:
        // helper: compute histogram of grayscale image
        std::vector<int> computeHistogram(const cv::Mat& input, std::optional<int> plot = {}) const;

        // helper: compute within-class variance for threshold t
        double computeBetweenClassVariance(const std::array<uint32_t,256>& cum_count,const std::array<uint64_t,256>& cum_sum ,int t, std::optional<int> profile = {}) const;
    };

}
