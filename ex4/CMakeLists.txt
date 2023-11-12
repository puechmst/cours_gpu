include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(managed managed.cu )
set_target_properties(managed PROPERTIES CUDA_ARCHITECTURES "75")