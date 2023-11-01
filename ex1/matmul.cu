
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCK_DIM (16)

__global__ void matmul(int lda, int ncola, float* a, int ncolb, float* b, float* c) {
	// get indices in the matrix
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	float s;
	if (i < lda && j < ncolb) {	// check validity of thread
		s = 0.0f;
		for (int k = 0; k < ncola; k++) // accumulate products
			s += a[i * ncola + k] * b[k * ncolb + j];
		c[i * ncolb + j] = s;
	}
}
__global__ void block_matmul(int lda, int ncola, float* a, int ncolb, float* b, float* c) {
	// get indices in the matrix
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	__shared__ float blockA[BLOCK_DIM][BLOCK_DIM];
	__shared__ float blockB[BLOCK_DIM][BLOCK_DIM];
	__shared__ float blockC[BLOCK_DIM][BLOCK_DIM];
	
	float s;

	// iterate over blocks of size BLOCK_DIM * BLOCK_DIM
	// compute block matrix product
	// store result.	
	if (i < lda && j < ncolb) {	// check validity of thread
}
}


void device_matmul(int lda, int ncola, float* a, int ncolb, float* b, float* c) {
	int nbx, nby;
	// compute required numvber of blocs in each direction
	nbx = (lda + BLOCK_DIM - 1) / BLOCK_DIM;
	nby = (ncolb + BLOCK_DIM - 1) / BLOCK_DIM;
	// allocate device memory
	float* da, * db, * dc;
	cudaMalloc(&da, lda * ncola * sizeof(float));
	cudaMalloc(&db, ncolb * ncola * sizeof(float));
	cudaMalloc(&dc, lda * ncolb * sizeof(float));
	cudaMemcpy(da, a, lda * ncola * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, ncolb * ncola * sizeof(float), cudaMemcpyHostToDevice);
	matmul <<< dim3(nbx, nby, 1), dim3(BLOCK_DIM, BLOCK_DIM, 1) >>> (lda, ncola, da, ncolb, db, dc);
	cudaDeviceSynchronize();
	cudaMemcpy(c, dc, lda * ncolb * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

void host_matmul(int lda, int ncola, float *   a, int ncolb, float*  b, float*  c) {
	float s;
	for(int i = 0 ; i < lda ; i++)
		for (int j = 0; j < ncolb; j++) {
			s = 0.0f;
			for (int k = 0; k < ncola; k++)
				s += a[i * ncola + k] * b[k * ncolb + j];
			c[i * ncolb + j] = s;
		}
}