#include<cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM (256)

__global__ void reduction(float *a, float *buffer, int n) {
    __shared__ float r[BLOCK_DIM];
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int nb = (n + BLOCK_DIM -1) / BLOCK_DIM;
    int step = BLOCK_DIM >> 1;
    if (i < n) 
        r[i] = a[i];
    else 
        r[i] = 0.0f;
    __syncthreads();
    while(step >= 1) {
        if(threadIdx.x < step)
           r[threadIdx.x] += r[threadIdx.x + step];
        __syncthreads();
        step >>= 1;
    }
    buffer[blockIdx.x] = r[0];
    if(i == 0) {
        if(nb > 1)
            reduction<<<nb, BLOCK_DIM,0,cudaStreamTailLaunch>>>(buffer,buffer,nb);
        else
            for(int k = 0 ; k < BLOCK_DIM ; k++)
                buffer[0] += r[i];
    }
}

float device_reduction(float *a, int n) {
    float *da;
    float *dbuffer;
    float *hbuffer;
    float res;
    int sz = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    hbuffer = new float[sz];
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&dbuffer, sz * sizeof(float));
    cudaMemcpy(da, a, n * sizeof(float),cudaMemcpyHostToDevice);
    reduction<<<sz, BLOCK_DIM>>>(da, dbuffer, n);
    cudaDeviceSynchronize();
    cudaMemcpy(hbuffer, dbuffer, sz * sizeof(float),cudaMemcpyDeviceToHost);
    res = hbuffer[0];
    delete[] hbuffer;
    cudaFree(da);
    cudaFree(dbuffer);
    return res;
}