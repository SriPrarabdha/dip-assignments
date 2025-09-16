#include "imageproc/binarize.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Gnuplot-iostream.h" 


using namespace std;

namespace imageproc {

    cv::Mat OtsuBinarizer::apply(const cv::Mat& img, double& optimalVariance) const {
        // TODO:
        // 1. compute histogram
        // 2. iterate over all thresholds t
        // 3. compute within-class variance
        // 4. pick threshold that minimizes it
        // 5. return thresholded binary image
        auto hist = OtsuBinarizer::computeHistogram(img);
        for(int i = 0; i<256; i++){
            cout<<i<<" - "<<hist[i]<<"\n";
        }
        return img.clone(); // placeholder
    }

    std::vector<int> OtsuBinarizer::computeHistogram(const cv::Mat& img) const {
        vector<int> hist(256, 0);

        cout<<img.rows<<" , "<<img.cols<<"\n";
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                uchar pixel = img.at<uchar>(i, j);  
                // std::cout << static_cast<int>(pixel) << "\n";
                hist[static_cast<int>(pixel)]++;
            }
        }

        vector<pair<int,int>> histData;
        for (int i = 0; i <= 255; i++) {
            histData.push_back({i, hist[i]});
        }

        // Plot with gnuplot
        Gnuplot gp;
        gp << "set title 'Image Histogram'\n";
        gp << "set xlabel 'Intensity'\n";
        gp << "set ylabel 'Frequency'\n";
        gp << "set style fill solid\n";
        gp << "plot '-' with boxes lc rgb 'blue' notitle\n";
        gp.send1d(histData);

        return hist;
    }

    double OtsuBinarizer::computeWithinClassVariance(const std::vector<int>& hist, int t) const {
        // TODO: compute σ_w² for threshold t
        return 0.0;
    }

}
