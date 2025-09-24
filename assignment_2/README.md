## Dependencies
- C++17 or newer
- CMake >= 3.16
- OpenCV >= 4.0

### linux
```
sudo apt update
sudo apt install build-essential cmake libopencv-dev
sudo apt-get install libboost-all-dev
sudo apt install -y gnuplot
```


### macOS
```
brew install cmake opencv gnuplot
```

## Build the project
```
rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
./apps/run_exp
```

### or just run this bash file

chmod +x run.sh   

./run.sh 1
./run.sh --filter box --input ../data/moon_noisy.png



