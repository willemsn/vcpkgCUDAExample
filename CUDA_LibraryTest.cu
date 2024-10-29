#include <stdio.h>
#include <stdlib.h>

#include "CUDA_DeviceFunctions.h"

__global__ void kernel (int x)
{
    printf ("GPU: %d\n", deviceA(x));
}

int main (void)
{
    kernel<<<1,1>>>(5);
    cudaDeviceSynchronize();
    
    return EXIT_SUCCESS;
}