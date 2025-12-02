#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>

void dumpCharacteristics() {
       size_t free, total;
       int ndevices;
       cudaDeviceProp prop;
       cudaGetDeviceCount(&ndevices);
       for(int device = 0 ; device < ndevices ; device++) {
            std::cout << "=============================================" << std::endl;
            cudaGetDeviceProperties(&prop, device);
            std::cout << "Device " << device << std::endl;
            std::cout << "Name " << prop.name << std::endl;
            std::cout << "Max threads per block : " << prop.maxThreadsPerBlock << std::endl;
            std::cout << "Warp size : " << prop.warpSize << std::endl;
            std::cout << "Shared memory per block : " << prop.sharedMemPerBlock << std::endl;
            std::cout << "Multiprocessor count : " << prop.multiProcessorCount << std::endl;
            std::cout << "Maximum bloc size in x coordinate : " << prop.maxThreadsDim[0] << std::endl;
            std::cout << "Maximum bloc size in y coordinate : " << prop.maxThreadsDim[1] << std::endl;
            std::cout << "Maximum bloc size in z coordinate : " << prop.maxThreadsDim[2] << std::endl;
            std::cout << "Maximum grid size in x coordinate : " << prop.maxGridSize[0] << std::endl;
            std::cout << "Maximum grid size in y coordinate : " << prop.maxGridSize[1] << std::endl;
            std::cout << "Maximum grid size in z coordinate : " << prop.maxGridSize[2] << std::endl;
            std::cout << "=============================================" << std::endl;
       }

       cudaMemGetInfo(&free, &total);
       std::cout << "Total memory : " << total << " Free memory : " << free << std::endl;

}