#include <stdio.h>
#include <iostream>

#define SIZE (100)
#define HSIZE (10)
#define NROW (100)
#define NCOL (100)
#define NRH (10)
#define NCH (10)
#define BSIZE (64)
#define BX (8)
#define BY (8)

void cpu_conv1d(int n, float *x, int p, float *h, float *y)
{
   // calcul du produit de convolution 1d sur CPU.
   for (int i = 0; i < n; i++)
   {
      float s = 0.0;
      for (int j = max(0, i - p + 1); j <= i; j++)
      {
         s += x[j] * h[i - j];
      }
      y[i] = s;
   }
}

void cpu_conv2d(int m, int n, float *x, int p, int q, float *h, float *y)
{
   // calcul du produit de convolution 2D sur CPU.
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         float s = 0.0;
         for (int k = max(0, i - p + 1); k <= i; k++)
         {
            for (int l = max(0, j - q + 1); l <= j; l++)
               s += x[k * n + l] * h[(i - k) * q + j - l];
         }
         y[i * n + j] = s;
      }
   }
}

__global__ void gpu_conv1d(int n, float *x, int p, float *h, float *y)
{
   int i = threadIdx.x + blockIdx.x * BSIZE;
   if (i < n)
   {
      float s = 0.0;
      for (int j = max(0, i - HSIZE + 1); j <= i; j++)
      {
         s += x[j] * h[i - j];
      }
      y[i] = s;
   }
} 

__global__ void gpu_conv2d(int m, int n, float *x, int p, int q, float *h, float *y)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;

   if (i < m && j < n)
    {
      float s = 0.0;
      for (int k = max(0, i - p + 1); k <= i; k++)
         {
            for (int l = max(0, j - q + 1); l <= j; l++)
               s += x[k * n + l] * h[(i - k) * q + j - l];
         }
      y[i * n + j] = s;
   }
} 

float test_conv1d() {
   float *x, *y, *h;
   float *dx, *dy, *dh;
   int nb;

   x = new float[SIZE];
   y = new float[SIZE];
   h = new float[HSIZE];

   for (int i = 0; i < SIZE; i++)
      x[i] = 1.0f;

   for (int i = 0; i < HSIZE; i++)
      h[i] = 1.0f;

   nb = (SIZE + BSIZE - 1) / BSIZE;
   cudaMalloc(&dx, SIZE * sizeof(float));
   cudaMalloc(&dy, SIZE * sizeof(float));
   cudaMalloc(&dh, HSIZE * sizeof(float));
   cudaMemcpy(dx, x, SIZE * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(dh, h, HSIZE * sizeof(float), cudaMemcpyHostToDevice);
   cpu_conv1d(SIZE, x, HSIZE, h, y);
   gpu_conv1d<<<nb, BSIZE>>>(SIZE, dx, HSIZE, dh, dy);
   cudaMemcpy(x, dy, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
   float err = 0.0f;
   for (int i = 0; i < SIZE; i++)
      err += fabsf(y[i] - x[i]);
   cudaFree(dx);
   cudaFree(dy);
   cudaFree(dh);
   delete[] x;
   delete[] y;
   delete[] h;
   return err;
}

float test_conv2d() {
   float *x, *y, *h;
   float *dx, *dy, *dh;
   int nbx, nby;

   x = new float[NROW * NCOL];
   y = new float[NROW * NCOL];
   h = new float[NRH * NCH];

   for (int i = 0; i < NROW * NCOL; i++)
      x[i] = 1.0f;

   for (int i = 0; i < NRH * NCH; i++)
      h[i] = 1.0f;

   nbx = (NROW + BX - 1) / BX;
   nby = (NCOL + BY - 1) / BY;
   cudaMalloc(&dx, NROW * NCOL * sizeof(float));
   cudaMalloc(&dy, NROW * NCOL * sizeof(float));
   cudaMalloc(&dh, NRH * NCH * sizeof(float));
   cudaMemcpy(dx, x, NROW * NCOL * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(dh, h, NRH * NCH * sizeof(float), cudaMemcpyHostToDevice);
   cpu_conv2d(NROW, NCOL, x, NRH, NCH, h, y);
   gpu_conv2d<<<dim3(nbx, nby, 1), dim3(BX, BY, 1)>>>(NROW, NCOL, dx, NRH, NCH, dh, dy);
   cudaMemcpy(x, dy, NROW * NCOL * sizeof(float), cudaMemcpyDeviceToHost);
   float err = 0.0f;
   for (int i = 0; i < NROW * NCOL; i++)
      err += fabsf(y[i] - x[i]);
   cudaFree(dx);
   cudaFree(dy);
   cudaFree(dh);
   delete[] x;
   delete[] y;
   delete[] h;
   return err;
}


int main(int argc, char *argv[])
{
   std::cout << test_conv1d() << '\t' << test_conv2d()<< std::endl;
   return 0;
}