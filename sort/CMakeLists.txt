include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(sort sort.cpp sort.cu "sort.h")
set_property(TARGET reduction PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --relocatable-device-code=true
                       -cudadevrt static
                       >)
set_target_properties(sort PROPERTIES CUDA_ARCHITECTURES "75")