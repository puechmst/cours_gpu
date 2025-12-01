cmake_minimum_required(VERSION 3.10)
project(add LANGUAGES CXX CUDA)
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#add_subdirectory(RK45)
add_executable(add main.cpp add.cu check.cu
        )
set_target_properties(add PROPERTIES CUDA_ARCHITECTURES "75")