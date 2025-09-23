#include "imageproc/transforms.hpp"
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;

namespace imageproc {

    // Clamp helper
    inline int clamp(int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    }

    // Bilinear interpolation helper
    inline cv::Vec3b bilinear_sample(const cv::Mat& img, float x, float y) {
        int x0 = (int)std::floor(x);
        int y0 = (int)std::floor(y);
        int x1 = clamp(x0 + 1, 0, img.cols - 1);
        int y1 = clamp(y0 + 1, 0, img.rows - 1);

        float wx = x - x0;
        float wy = y - y0;

        cv::Vec3b p00 = img.at<cv::Vec3b>(clamp(y0,0,img.rows-1), clamp(x0,0,img.cols-1));
        cv::Vec3b p10 = img.at<cv::Vec3b>(clamp(y0,0,img.rows-1), x1);
        cv::Vec3b p01 = img.at<cv::Vec3b>(y1, clamp(x0,0,img.cols-1));
        cv::Vec3b p11 = img.at<cv::Vec3b>(y1, x1);

        cv::Vec3b result;
        for (int c = 0; c < 3; c++) {
            float val =
                (1 - wx) * (1 - wy) * p00[c] +
                wx * (1 - wy) * p10[c] +
                (1 - wx) * wy * p01[c] +
                wx * wy * p11[c];
            result[c] = (uchar)clamp((int)std::round(val), 0, 255);
        }
        return result;
    }

    // Rotate an image by a given angle (degrees, around center) using bilinear interpolation
    cv::Mat transforms::rotate(const cv::Mat& input, int angle) {
        float theta = angle * M_PI / 180.0f;
        
        int w = input.cols;
        int h = input.rows;
        int cx = w / 2;
        int cy = h / 2;
        
        cv::Mat output(h, w, input.type(), cv::Scalar::all(0));
        // cout<<"check 0\n";

        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        // cout<<"check 1";

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // inverse mapping
                float x_shift = x - cx;
                float y_shift = y - cy;

                float src_x = cos_t * x_shift + sin_t * y_shift + cx;
                float src_y = -sin_t * x_shift + cos_t * y_shift + cy;

                if (src_x >= 0 && src_x < w - 1 && src_y >= 0 && src_y < h - 1) {
                    output.at<cv::Vec3b>(y, x) = bilinear_sample(input, src_x, src_y);
                }
            }
        }

        return output;
    }

    // Bilinear interpolation upsampling
    cv::Mat transforms::interpolate_bilinear(const cv::Mat& input, int factor) {
        int new_h = input.rows * factor;
        int new_w = input.cols * factor;
        cv::Mat output(new_h, new_w, input.type());

        for (int y = 0; y < new_h; y++) {
            float src_y = (float)y / factor;
            for (int x = 0; x < new_w; x++) {
                float src_x = (float)x / factor;
                output.at<cv::Vec3b>(y, x) = bilinear_sample(input, src_x, src_y);
            }
        }
        return output;
    }

    // Upsampling (nearest neighbor for baseline, but you want bilinear)
    cv::Mat transforms::upsample(const cv::Mat& input, int factor) {
        return interpolate_bilinear(input, factor);
    }


}
