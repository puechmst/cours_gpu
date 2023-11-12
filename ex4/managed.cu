#include<cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM (256)
#define SIZE (10000)

__device__ __managed__ float c[SIZE];

__global__ void add(float *a, float *b, int n) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {
    float *a,*b;
    int ns  = (SIZE + BLOCK_DIM -1) / BLOCK_DIM;
    cudaMallocManaged(&a, SIZE * sizeof(float));
    cudaMallocManaged(&b, SIZE * sizeof(float));
    for(int i = 0 ; i < SIZE ; i++) {
        a[i] = 1.0f;
        b[i] = (float)i;
    }
    add<<<ns, BLOCK_DIM>>>(a,b,SIZE);
    cudaDeviceSynchronize();
    float err = 0.0f;
    for(int i = 0 ; i < SIZE ; i++) {
        err += fabsf(b[i]+1.0f-c[i]);
    }
    std::cout << "Erreur : " << err << std::endl;
    cudaFree(b);
    cudaFree(a);
}