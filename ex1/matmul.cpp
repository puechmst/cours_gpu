#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>


void dumpCharacteristics() {
       size_t free, total;
       int attr;
       int ndevices;
       int cptminor, cptmajor;
       cudaGetDeviceCount(&ndevices);
       for(int device = 0 ; device < ndevices ; device++) {
            std::cout << "=============================================" << std::endl;
            std::cout << "Device " << device << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxThreadsPerBlock,device);
            std::cout << "Max threads per block : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrWarpSize,device);
            std::cout << "Warp size : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxBlockDimX,device);
            std::cout << "Maximum bloc size in x coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxBlockDimY,device);
            std::cout << "Maximum bloc size in y coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxBlockDimZ,device);
            std::cout << "Maximum bloc size in z coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimX,device);
            std::cout << "Maximum grid size in x coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimY,device);
            std::cout << "Maximum grid size in y coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimZ,device);
            std::cout << "Maximum grid size in z coordinate : " << attr << std::endl;
            cudaDeviceGetAttribute(&attr, cudaDevAttrMultiProcessorCount,device);
            std::cout << "Number of multiprocessors : " << attr << std::endl;
            cudaDeviceGetAttribute(&cptmajor, cudaDevAttrComputeCapabilityMajor,device);
            cudaDeviceGetAttribute(&cptminor, cudaDevAttrComputeCapabilityMinor, device);
            std::cout << "Compute capability : " << cptmajor << '.' << cptminor << std::endl;
            std::cout << "=============================================" << std::endl;
       }

       cudaMemGetInfo(&free, &total);
       std::cout << "Total memory : " << total << " Free memory : " << free << std::endl;

}

int main(int argc, char* argv[]) {
    dumpCharacteristics();
	return 0;
}