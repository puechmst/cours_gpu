
#include<cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
// la taille d'un bloc doit être un multiple de 2
#define BLOCK_DIM (256)

// note: un noyau ne peut pas retourner de valeur. Il faut la récupérer à l'aide d'une fonction spéciale
__device__ float res;

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
    c[blockIdx.x] = r[0];
    // final reduction
    __syncthreads();
    if(i == 0) { 
        res = 0.0f;
        for(int k = 0 ; k < gridDim.x ; k++)
            res += c[k];
    }
 }

void test_dot(int n) {
    float *a, *b, *c;
    float *da, *db, *dc;
    float dot[1];
    a = new float[n];
    b = new float[n];
    int sz = (n+BLOCK_DIM -1)/BLOCK_DIM;
    c = new float[sz];
    for(int i = 0 ; i < n ; i++) {
        a[i] = 1.0/(float)(i+1);
        b[i] = (float)(i+1);
    }
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, sz * sizeof(float));
    cudaMemcpy(da, a, n * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * sizeof(float),cudaMemcpyHostToDevice);
    dot_product<<<sz,BLOCK_DIM>>>(da, db, dc, n);
    cudaMemcpyFromSymbol(dot, res, sizeof(float));
    cudaMemcpy(c, dc, sz * sizeof(float),cudaMemcpyDeviceToHost);
    float r = 0.0f;
    float r1 = 0.0f;
    for(int i = 0 ; i < n ; i++) 
            r += a[i] * b[i];
    for(int i = 0 ; i < sz ; i++)
            r1 += c[i];
    std::cout << *dot << '\t' << r1 << '\t' << r <<std::endl;
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete []a;
    delete []b;
    delete []c;
}
 