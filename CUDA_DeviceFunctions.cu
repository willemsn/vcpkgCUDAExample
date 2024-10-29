#include "CUDA_DeviceFunctions.h"

__device__ int deviceA(int x)
{
    return x * x;
}
