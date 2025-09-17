## Dependencies
- C++17 or newer
- CMake >= 3.16
- OpenCV >= 4.0

### Ubuntu
sudo apt update
sudo apt install build-essential cmake libopencv-dev
sudo apt-get install libboost-all-dev


### macOS
brew install cmake opencv


rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
./apps/run_exp


chmod +x run.sh   # make it executable

./run.sh 1
./run.sh --filter box --input ../data/moon_noisy.png


#include <vector>
#include <array>
#include <optional>
#include <chrono>
#include <iostream>
#include <cstdint>

