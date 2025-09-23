cmake_minimum_required(VERSION 3.16)
project(imageproc LANGUAGES CXX)

# Require C++17 (or newer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)

find_program(GNUPLOT_EXECUTABLE gnuplot)

if(NOT GNUPLOT_EXECUTABLE)
    message(FATAL_ERROR "gnuplot executable not found")
endif()

add_library(imageproc
    src/filter.cpp
    src/binarize.cpp
    src/io.cpp
    src/transforms.cpp
)


target_include_directories(imageproc
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/gnuplot-iostream
        ${Boost_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(imageproc
    PUBLIC 
        ${OpenCV_LIBS}
        Boost::iostreams
        Boost::system
        Boost::filesystem
)

# Add apps subdirectory
target_compile_options(imageproc PRIVATE -O3 -march=native)
add_subdirectory(apps)
