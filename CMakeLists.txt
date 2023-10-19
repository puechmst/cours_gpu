cmake_minimum_required(VERSION 3.8)
project(coursgpu LANGUAGES CXX CUDA)
add_executable(gpu main.cpp add.cu check.cu)
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "50;75")