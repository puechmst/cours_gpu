include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(matmul matmul.cpp matmul.cu)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "75")