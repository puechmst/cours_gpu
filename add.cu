#include<stdlib.h>
#include<stdio.h>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include "add.h"
#define THREADS_PER_BLOCK (512)

__global__ void add_kernel(int n, float *u, float *v, float *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = u[i] + v[i];
    }
}

__global__ void setup_rnd_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
 
    curand_init(1234, id, 0, &state[id]);
}

__global__ void add_init_kernel(curandState *state,
                                int n,
                                float *u)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        /* Copy state to local memory for efficiency */
        curandState localState = state[i];
        u[i] = curand_uniform(&localState); 
        /* Copy state back to global memory */
        state[i] = localState;
    }
}

void rnd_init(int n, float *u) {
    for (int i = 0 ; i < n ; i++)
        u[i] = drand48();
}

int add_unified(int n) {
    float *a, *b, *c;
    curandState *rnd_states;
    int sz;
    int n_blocks;
    int b_pass;
    sz = n * sizeof(float);
    cudaMallocManaged((void **)&a, sz);
    cudaMallocManaged((void **)&b, sz);
    cudaMallocManaged((void **)&c, sz);
    cudaMallocManaged((void **)&rnd_states, n * sizeof(curandState));

    n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    //rnd_init(n, a);
    //rnd_init(n, b);
    setup_rnd_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(rnd_states );
    cudaDeviceSynchronize();
    add_init_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(rnd_states, n, a );
    cudaDeviceSynchronize();
    add_init_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(rnd_states, n, b );
    cudaDeviceSynchronize();
    add_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(n, a, b, c);
    cudaDeviceSynchronize();
     // verification
    b_pass = 1;
    for(int i = 0 ; i < n ; i++) {
       if (c[i] != a[i] + b[i]) {
            b_pass = 0;
            break;
       }
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(rnd_states);
    return b_pass;
}

int  launch_and_test(int n) {
    float *a,*b,*c;
    float *da,*db,*dc;
    int n_blocks;
    int b_pass;
    int sz;

    sz = n * sizeof(float);
    a = (float *)malloc(sz);
    b = (float *)malloc(sz);
    c = (float *)malloc(sz);

    cudaMalloc((void **)&da, sz);
    cudaMalloc((void **)&db, sz);
    cudaMalloc((void **)&dc, sz);

    // remplissage aleatoire
    rnd_init(n, a); 
    rnd_init(n, b);

    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);

    n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(n, da, db, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // verification
    b_pass = 1;
    for(int i = 0 ; i < n ; i++) {
       if (c[i] != a[i] + b[i]) {
            b_pass = 0;
            break;
       }
    }

    free(a);
    free(b);
    free(c);

    return b_pass;
}