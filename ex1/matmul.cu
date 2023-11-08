
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCK_DIM (8)

__global__ void matmul(int lda, int ncola, float *a, int ncolb, float *b, float *c)
{
	// indices de l'élément calculé
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	float s;
	if (i < lda && j < ncolb)
	{ // test de l'appartenance à la matrice
		s = 0.0f;
		for (int k = 0; k < ncola; k++) // somme des produits
			s += a[i * ncola + k] * b[k * ncolb + j];
		c[i * ncolb + j] = s;
	}
}

__global__ void block_matmul(int lda, int ncola, float *a, int ncolb, float *b, float *c)
{
	// indice de début des blocs A , B et C
	int ia = blockIdx.x * ncola * BLOCK_DIM;
	int ib = blockIdx.y * BLOCK_DIM;
	int ic = blockIdx.x * ncolb * BLOCK_DIM + blockIdx.y * BLOCK_DIM;
	// pas en A et B
	int sa = BLOCK_DIM;
	int sb = ncolb * BLOCK_DIM;
	__shared__ float blockA[BLOCK_DIM][BLOCK_DIM];
	__shared__ float blockB[BLOCK_DIM][BLOCK_DIM];
	int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
	float s = 0.0f;
	for (int k = 0; k < ncola; k += BLOCK_DIM, ia += sa, ib += sb)
	{
		// chargement d'un élément en mémoire partagée
		if (i < lda)
			blockA[threadIdx.x][threadIdx.y] = a[ia + threadIdx.y + threadIdx.x * ncola];
		else
			blockA[threadIdx.x][threadIdx.y] = 0.0f;
		if (j < ncolb)
			blockB[threadIdx.x][threadIdx.y] = b[ib + threadIdx.y + threadIdx.x * ncolb];
		else
			blockB[threadIdx.x][threadIdx.y] = 0.0f;
		__syncthreads(); // point de synchronisation pour s'assurer du chargement complet des blocs
		// calcul du produit matriciel
		int rem = min(BLOCK_DIM, ncola - k);
		for (int l = 0; l < rem; l++)
			s += blockA[threadIdx.x][l] * blockB[l][threadIdx.y];
		__syncthreads(); // point de synchronisation pour les calculs
	}
	if ((i < lda) && (j < ncolb))
		c[ic + threadIdx.y + threadIdx.x * ncolb] = s;
}

void device_matmul(int lda, int ncola, float *a, int ncolb, float *b, float *c)
{
	int nbx, nby;
	// compute required number of blocs in each direction
	nbx = (lda + BLOCK_DIM - 1) / BLOCK_DIM;
	nby = (ncolb + BLOCK_DIM - 1) / BLOCK_DIM;
	// allocate device memory
	float *da, *db, *dc;
	cudaMalloc(&da, lda * ncola * sizeof(float));
	cudaMalloc(&db, ncolb * ncola * sizeof(float));
	cudaMalloc(&dc, lda * ncolb * sizeof(float));
	cudaMemcpy(da, a, lda * ncola * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, ncolb * ncola * sizeof(float), cudaMemcpyHostToDevice);
	block_matmul<<<dim3(nbx, nby, 1), dim3(BLOCK_DIM, BLOCK_DIM, 1)>>>(lda, ncola, da, ncolb, db, dc);
	cudaDeviceSynchronize();
	cudaMemcpy(c, dc, lda * ncolb * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

void host_matmul(int lda, int ncola, float *a, int ncolb, float *b, float *c)
{
	float s;
	for (int i = 0; i < lda; i++)
		for (int j = 0; j < ncolb; j++)
		{
			s = 0.0f;
			for (int k = 0; k < ncola; k++)
				s += a[i * ncola + k] * b[k * ncolb + j];
			c[i * ncolb + j] = s;
		}
}
