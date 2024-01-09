#include<stdio.h>
#include<iostream>
#include<random>
#include<math.h>
#include"reduction.h"
#include<cuda_runtime.h>
#include <device_launch_parameters.h>

#define SIZE (512045)

int main(int argc, char *argv[]) {
    float *a = new float[SIZE];
    for(int i  = 0 ; i < SIZE ; i++)
        a[i] = 1.0f;
    std::cout << device_reduction(a,SIZE) << std::endl;
    delete[] a;
    return 0;
}