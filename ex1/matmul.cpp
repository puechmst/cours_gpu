#include<stdio.h>
#include<iostream>
#include<iomanip>
#include<random>
#include<math.h>
#include"matmul.h"
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include<chrono>


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

void random_matrix(int lda, int ncol, float* a) {
   // for debugging purpose, seed has a fixed value
    std::mt19937 gen(5);
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    for (int i = 0 ; i < lda * ncol ; i++)
    // random generator from c++ standard library
        a[i] = dist(gen);
}

int main(int argc, char* argv[]) {
    dumpCharacteristics();
    int lda = 4096;
    int ncola = 256;
    int ncolb = 4096;

    float *a = new float[lda * ncola];
    float *b = new float[ncola * ncolb];
    float *c = new float[lda * ncolb];
    float *d = new float[lda * ncolb];
    random_matrix(lda, ncola, a);
    random_matrix(ncola, ncolb, b);
    double err;
    std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();
    device_matmul(lda, ncola, a, ncolb, b, c);
    std::chrono::time_point<std::chrono::system_clock> eps1 =
        std::chrono::system_clock::now();
    host_matmul(lda, ncola, a, ncolb, b, d);
    std::chrono::time_point<std::chrono::system_clock> eps2 =
        std::chrono::system_clock::now();
    double dsize = (double)(lda) * (double) ncola * (double) ncolb;
    double gpuflops = 2.0e6 * dsize  / (double) std::chrono::duration_cast<std::chrono::microseconds>(eps1 - now).count();
    
    double cpuflops = 2.0e6 * dsize  / (double) std::chrono::duration_cast<std::chrono::microseconds>(eps2 - eps1).count();
    std::cout << "GPU Mflops " << gpuflops/1000000 << std::endl;
    std::cout << "CPU Mflops " << cpuflops/1000000 << std::endl;
    err = 0.0;
    double de;
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < ncolb; j++) {
            de = fabs((double)(d[i * ncolb + j]) - (double)(c[i * ncolb + j]));
            if (de > err) 
                err = de;
        }
    }
    std::cout << "Erreur " << err  << std::endl;
    delete []a;
    delete []b;
    delete []c;
    delete []d;
	return 0;
}