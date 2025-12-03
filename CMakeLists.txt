cmake_minimum_required(VERSION 3.18)
project(icp  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
SET(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(.)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

add_library(data_io SHARED src/data_io.cpp)
target_include_directories(data_io
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)

add_library(icp_cuda STATIC src/icp_cuda.cu)
target_include_directories(icp_cuda
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(icp_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

add_executable(registration src/main.cpp)
target_include_directories(registration
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(registration
    PRIVATE
        data_io
        icp_cuda
)
