cmake_minimum_required(VERSION 3.16)
project(imageproc LANGUAGES CXX)

# Require C++17 (or newer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(Gnuplot REQUIRED COMPONENTS iostream)

# Make library from src/*.cpp
add_library(imageproc
    src/filter.cpp
    src/binarize.cpp
    src/io.cpp
)

# Public headers are in include/
target_include_directories(imageproc
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Link OpenCV to our library
target_link_libraries(imageproc
    PUBLIC ${OpenCV_LIBS}
)

# Add apps subdirectory
add_subdirectory(apps)
