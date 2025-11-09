cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CNNExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/aredjil/opt/libtorch")  

find_package(Torch REQUIRED)

# Path to your data in the source tree
set(DATA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/data)

# Path where you want it in the build tree
set(DATA_BUILD_DIR ${CMAKE_BINARY_DIR}/data)

# Copy the data folder recursively
file(COPY ${DATA_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
add_executable(cnn src/main.cpp)
target_link_libraries(cnn "${TORCH_LIBRARIES}")
set_property(TARGET cnn PROPERTY CXX_STANDARD 17)
