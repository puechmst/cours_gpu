include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(reduction reduction.cpp reduction.cu "reduction.h")
target_compile_options(reduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --relocatable-device-code=true
                       -cudadevrt
                       -cudart
                       >)
set_target_properties(reduction PROPERTIES CUDA_ARCHITECTURES "75")