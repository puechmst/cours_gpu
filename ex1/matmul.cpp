#include<stdio.h>
#include<iostream>
#include<random>
#include<math.h>
#include"matmul.h"
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

void random_matrix(int lda, int ncol, float* a) {
   // for debugging purpose, seed has a fixed value
    std::mt19937 gen(5);

    for (int i = 0 ; i < lda * ncol ; i++)
    // random generator from c++ standard library
        a[i] = (float)((double)gen()/(double)(gen.max()));
}

int main(int argc, char* argv[]) {
    dumpCharacteristics();
    int lda = 1000;
    int ncola = 1000;
    int ncolb = 1000;

    float *a = new float[lda * ncola];
    float *b = new float[ncola * ncolb];
    float *c = new float[lda * ncolb];
    float *d = new float[lda * ncolb];
    random_matrix(lda, ncola, a);
    random_matrix(ncola, ncolb, b);
    float err;
    device_matmul(lda, ncola, a, ncolb, b, c);
    host_matmul(lda, ncola, a, ncolb, b, d);
    err = 0.0;
    float de;
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < ncolb; j++) {
            de = (d[i * ncolb + j] - c[i * ncolb + j]);
            err += de * de;
        }
    }
    std::cout << "Erreur relative: " << sqrt(err)/(float)(lda*ncolb)<< std::endl;
    delete a;
    delete b;
    delete c;
    delete d;
	return 0;
}