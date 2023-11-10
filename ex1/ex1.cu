#include<cuda_runtime.h>
#include <device_launch_parameters.h>


void host_matmul(int lda, int ncol, float* a, int ldb, float* b, float* res) {
	double s;

	for (int i = 0 ; i < lda; i++) {
		for (int j = 0 ; j < k ; j++) {
			s = 0.0;
			for (int k = 0; k < ldb; k++)
				s += a[i * lda + k] * b[k * ldb + j];
		}
		c[i * lda + j] = s;
	}
}

__global__ void device_matmul(int lda, int k, float* a, int ldb, float* b, float* res) {	

	
}