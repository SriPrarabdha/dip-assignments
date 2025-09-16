#pragma once
#include <opencv2/core.hpp>

namespace imageproc {

    class OtsuBinarizer {
    public:
        // apply Otsu's thresholding on input image
        cv::Mat apply(const cv::Mat& input, double& optimalVariance) const;

    private:
        // helper: compute histogram of grayscale image
        std::vector<int> computeHistogram(const cv::Mat& input) const;

        // helper: compute within-class variance for threshold t
        double computeWithinClassVariance(const std::vector<int>& hist, int t) const;
    };

}
