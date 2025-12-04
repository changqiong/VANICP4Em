cmake_minimum_required(VERSION 3.18)
project(vanicp  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
SET(CMAKE_BUILD_TYPE "Release")

find_package(CUDAToolkit REQUIRED)
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(.)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES 75)


add_library(io SHARED src/io.cpp)
target_include_directories(io
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)

add_library(icp STATIC src/vanicp.cu)
target_include_directories(icp
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)
target_link_libraries(icp
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

add_executable(vanicp src/main.cpp)
target_include_directories(vanicp
    PUBLIC
        ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(vanicp
    PRIVATE
        io
        icp
)
