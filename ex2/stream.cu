
#include<cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <future>
// la taille d'un bloc doit être un multiple de 2
#define BLOCK_DIM (256)

__global__ void dot_product(float *a, float *b, float *c, int  n) {
    __shared__ float r[BLOCK_DIM];
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int step = BLOCK_DIM >> 1;
    // chargement des données
    if(i < n) {
        r[threadIdx.x] = a[i] * b[i];
    } else
        r[threadIdx.x] = 0.0f;
    __syncthreads();
    // calcul de la somme
    while(step >= 1) {
        if(threadIdx.x < step)
           r[threadIdx.x] += r[threadIdx.x + step];
        __syncthreads();
        step >>= 1;
    }
    if(threadIdx.x == 0)
        c[blockIdx.x] = r[0];
 }

 float host_dot(float *a, float *b, int n) {
    float r = 0.0f;
    for(int i = 0 ; i < n ; i++)
        r += a[i] * b[i];
    return r;
 }

float device_dot(float *a, float *b, int n,cudaStream_t stream) {
    float *ha, *hb, *hc;
    float *da, *db, *dc;
    float dot = 0.0f;
    // allocation de mémoire verrouillée
    cudaMallocHost(&ha, n * sizeof(float));
    cudaMallocHost(&hb, n * sizeof(float));
    int sz = (n+BLOCK_DIM -1)/BLOCK_DIM;
    cudaMallocHost(&hc,sz * sizeof(float));
    // copie asynchrone en mémoire verrouillée
    cudaMemcpyAsync(ha, a, n * sizeof(float), cudaMemcpyHostToHost, stream);
    cudaMemcpyAsync(hb, b, n * sizeof(float), cudaMemcpyHostToHost,stream);
    // allocation sur le GPU
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, sz * sizeof(float));
    // copie asynchrone vers le GPU 
    cudaMemcpyAsync(da, ha, n * sizeof(float),cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(db, hb, n * sizeof(float),cudaMemcpyHostToDevice, stream);
    // appel du noyau sur le flux
    dot_product<<<sz,BLOCK_DIM,0, stream>>>(da, db, dc, n);
    // copie asynchrone vers le CPU
    cudaMemcpyAsync(hc, dc, sz * sizeof(float),cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // réduction
     for(int i = 0 ; i < sz ; i++)
            dot += hc[i];
    // libération de la mémoire GPU
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    // libération de la mémoire CPU
    cudaFreeHost(ha);
    cudaFreeHost(hb);
    cudaFreeHost(hc);
    return dot;
}

void dot_async(int n, int ntasks) {
    float *a, *b;
    std::future<float> *dot = new std::future<float>[ntasks];
    cudaStream_t *streams = new cudaStream_t[ntasks];
    a = new float[n];
    b = new float[n];
    for(int i = 0 ; i < n ; i++) {
        a[i] = 1.0f/(float)(i+1);
        b[i] = (float)(i+1);
    }
   
    float r = 0.0f;
    for(int i = 0 ; i < n ; i++) 
            r += a[i] * b[i];
    for(int tsk = 0 ; tsk < ntasks ; tsk++) {
        cudaStreamCreate(streams+tsk);
        dot[tsk] = std::async(std::launch::async, device_dot, a, b, n, streams[tsk]);
    }
    cudaDeviceSynchronize();
    for(int tsk = 0 ; tsk < ntasks ; tsk++) {
        dot[tsk].wait();
        cudaStreamDestroy(streams[tsk]);
        std::cout  << tsk << '\t' << r << '\t' << dot[tsk].get() <<std::endl;
    }

    delete[] streams;
    delete[] dot;
    delete[] a;
    delete[] b;
}
 