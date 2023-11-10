include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(gpu_ex1 gpu_ex1.cpp ex1.cu)
set_target_properties(gpu_ex1 PROPERTIES CUDA_ARCHITECTURES "50;75")