#include <iostream>
#include <imageproc/filter.hpp>
#include <imageproc/binarize.hpp>
#include <imageproc/io.hpp>
using namespace std;

int main() {
    // Load noisy image
    auto img = imageproc::io::loadImage("../ip_images/moon_noisy.png");
    imageproc::io::showImage("image", img);
    cout<<"done\n";

    imageproc::OtsuBinarizer binarizer;
    double opt_var = 0.12;
    binarizer.apply(img, opt_var);        
    // Try different filter sizes
    // for (int m : {5, 29, 129}) {
    //     imageproc::BoxFilter bf(m);
    //     auto blurred = bf.apply(img);

    //     // Apply Otsu's method
    //     imageproc::OtsuBinarizer ob;
    //     double variance = 0.0;
    //     auto binary = ob.apply(blurred, variance);

    //     std::cout << "Filter size " << m << " → optimal variance = " << variance << "\n";

    //     // Save or show results
    //     imageproc::io::saveImage("out_blur_" + std::to_string(m) + ".png", blurred);
    //     imageproc::io::saveImage("out_bin_" + std::to_string(m) + ".png", binary);
    // }

    return 0;
}
