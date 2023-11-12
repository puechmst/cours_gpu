#include<cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM (256)

__global__ void reduction(float *a, float *b, int n) {
    __shared__ float r[BLOCK_DIM];
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int nb = (n + BLOCK_DIM -1) / BLOCK_DIM;
    int step = BLOCK_DIM >> 1;
    if (i < n) 
        r[threadIdx.x] = a[i];
    else 
        r[threadIdx.x] = 0.0f;
    __syncthreads();
    while(step >= 1) {
        if(threadIdx.x < step)
           r[threadIdx.x] += r[threadIdx.x + step];
        __syncthreads();
        step >>= 1;
    }
    if(threadIdx.x == 0)
        b[blockIdx.x] = r[0];
    __syncthreads();
    if(i == 0 && nb > 1) 
            reduction<<<nb, BLOCK_DIM,0,cudaStreamTailLaunch>>>(b,b,nb);
}

float device_reduction(float *a, int n) {
    float *da;
    float *dbuffer;
    float hbuffer;
    float res;
    int sz = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&dbuffer, sz * sizeof(float));
    cudaMemcpy(da, a, n * sizeof(float),cudaMemcpyHostToDevice);
    reduction<<<sz, BLOCK_DIM>>>(da, dbuffer, n);
    cudaDeviceSynchronize();
    cudaMemcpy(&hbuffer, dbuffer, sizeof(float),cudaMemcpyDeviceToHost);
    res = hbuffer;
    cudaFree(da);
    cudaFree(dbuffer);
    return res;
}