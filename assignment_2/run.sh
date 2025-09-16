#!/bin/bash

# exit if any command fails
set -e

# go into build dir (create if missing)
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake ..
    cd ..
fi

# build project
cmake --build build

# run the executable with arguments passed to this script
./build/apps/run_exp "$@"
