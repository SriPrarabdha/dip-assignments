#include "imageproc/binarize.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "gnuplot-iostream.h" 
#include <optional>
#include <numeric>

using namespace std;

namespace imageproc {

    cv::Mat OtsuBinarizer::apply(const cv::Mat& img, double& optimalVariance, std::optional<int> profile) const {
        bool profile_flag = profile.has_value();

        std::chrono::high_resolution_clock::time_point start;
        if (profile_flag) {
            start = std::chrono::high_resolution_clock::now();
        }

        // 1. compute histogram
        auto hist = OtsuBinarizer::computeHistogram(img);

        // 2. get ready for computation 
        std::array<uint32_t, 256> cum_count{};
        std::array<uint64_t, 256> cum_sum{};

        // Running totals
        uint32_t running_count = 0;
        uint64_t running_sum = 0;

        for (int i = 0; i < 256; i++) {
            uint64_t pixel = static_cast<uint64_t>(i);
            uint64_t h = static_cast<uint64_t>(hist[i]);

            running_count += static_cast<uint32_t>(h);
            running_sum   += h * pixel;

            cum_count[i] = running_count;
            cum_sum[i]   = running_sum;
        }

        uint32_t total_pixels = cum_count[255];

        double max_var = -1e6;
        int opt_thres = 0;
        // double total_mean = static_cast<double>(cum_sum[255]) / total_pixels;

        // 3. iterate over all thresholds t
        for(int t = 0; t<255; t++){
            uint32_t count0 = cum_count[t];
            uint64_t sum0   = cum_sum[t];

            // Class 1 (> t)
            uint32_t count1 = total_pixels - count0;
            uint64_t sum1   = cum_sum[255] - sum0;

            if (count0 == 0 || count1 == 0) {
                continue;
            }

            double mean0 = static_cast<double>(sum0) / count0;
            double mean1 = static_cast<double>(sum1) / count1;

            double w0 = static_cast<double>(count0) / total_pixels;

            double var = w0 * (1.0-w0) * (mean0 - mean1) * (mean0 - mean1);
            // cout<<t<<" , "<<var<<"\n";
            if (var>max_var) {max_var = var; opt_thres = t;}
        }

        cout<<opt_thres<<"\n";

        // 5. return thresholded binary image
        cv::Mat output_img;
        cv::threshold(img, output_img, opt_thres, 255.0, cv::THRESH_BINARY);

        if (profile_flag) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Otsu Thresholding took " << duration.count() << " ms." << std::endl;
        }
        return output_img; 
    }

    std::vector<int> OtsuBinarizer::computeHistogram(const cv::Mat& img, std::optional<int> plot) const {
        vector<int> hist(256, 0);

        // cout<<img.rows<<" , "<<img.cols<<"\n";

        // for (int i = 0; i < img.rows; i++) {
        //     for (int j = 0; j < img.cols; j++) {
        //         uchar pixel = img.at<uchar>(i, j);  
        //         // std::cout << static_cast<int>(pixel) << "\n";
        //         hist[pixel]++;
        //     }
        // }

        const uchar* p = img.ptr<uchar>(0);
        size_t total = img.rows * img.cols;
        for (size_t i = 0; i < total; i++) {
            hist[p[i]]++;
        }

        if(plot.has_value()){
            vector<pair<int,int>> histData;

            for (int i = 0; i < 256; i++) {
                histData.push_back({i, hist[i]});
            }

            // Plot histogram
            Gnuplot gp;
            gp << "set title 'Grayscale Histogram'\n";
            gp << "set xlabel 'Intensity'\n";
            gp << "set ylabel 'Frequency'\n";
            gp << "set style fill solid\n";
            gp << "plot '-' with boxes lc rgb 'blue' notitle\n";
            gp.send1d(histData);
        }
        return hist;
    }

    double OtsuBinarizer::computeBetweenClassVariance(const std::array<uint32_t,256>& cum_count,const std::array<uint64_t,256>& cum_sum , int t, std::optional<int> profile) const {
        bool profile_flag = profile.has_value();

        std::chrono::high_resolution_clock::time_point start;
        if (profile_flag) {
            start = std::chrono::high_resolution_clock::now();
        }

        uint32_t total_pixels = cum_count[255];
        if (total_pixels == 0) return 0.0; // safeguard

        // Class 0 (<= t)
        uint32_t count0 = cum_count[t];
        uint64_t sum0   = cum_sum[t];

        // Class 1 (> t)
        uint32_t count1 = total_pixels - count0;
        uint64_t sum1   = cum_sum[255] - sum0;

        if (count0 == 0 || count1 == 0) {
            return 0.0; 
        }

        double mean0 = static_cast<double>(sum0) / count0;
        double mean1 = static_cast<double>(sum1) / count1;

        double w0 = static_cast<double>(count0) / total_pixels;
        double w1 = 1.0 - w0;

        double var_b = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);

        if (profile_flag) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            std::cout << "Operation took " << duration.count() << " ns." << std::endl;
        }

        return var_b;
    }

}
