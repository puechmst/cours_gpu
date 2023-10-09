cmake_minimum_required(VERSION 3.2)
project(coursgpu LANGUAGES CXX CUDA)

add_executable(gpu main.cpp add.cu)