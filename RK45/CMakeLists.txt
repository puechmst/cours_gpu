include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(simu simu.cpp rk45.cu )
set_property(TARGET simu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(simu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --relocatable-device-code=true
                       -cudadevrt static
                       >)
set_target_properties(simu PROPERTIES CUDA_ARCHITECTURES "75")