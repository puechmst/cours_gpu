include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(stream stream.cpp stream.cu "stream.h")
set_target_properties(stream PROPERTIES CUDA_ARCHITECTURES "75")