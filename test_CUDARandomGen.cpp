#include <iostream>
#include <optional>
#include <tuple>
#include <cmath>

#include "CUDARandomKernel.h"

int main(int argc, char* argv[])
{
  // Include some C++ 17 tests
  std::cout << "C++ version: " << __cplusplus << std::endl;

  std::optional<int> opt = 42;
  std::cout << opt.value() << std::endl;

  std::tuple<int, float> aTuple = {42, 3.14159F};
  auto [a, b] = aTuple;
  std::cout << "a = " << a << ", b = " << b << std::endl;

  // Then, try CUDA
  size_t i;

  curandGenerator_t gen;
  float *devPRNVals, *hostData, *devResults, *hostResults;
    
  // Create pseudo-random number generator
  // CURAND_CALL(
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
  // Set the seed --- not sure how we'll do this yet in general
  // CURAND_CALL(
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  // start with 15k "particles"
  int numParticles = 15000;
    
  // repeat with ever increasing numbers of particles
  for (int r=0; r<15; r++) {

    // Allocate numParticle * 3 floats on host
    int n = numParticles * 3;

    hostResults = (float *)calloc(n, sizeof(float));
    
    // Allocate n floats on device to hold random numbers
    // CUDA_CALL();
    cudaMalloc((void **)&devPRNVals, n*sizeof(float));
    // CUDA_CALL();
    cudaMalloc((void **)&devResults, n*sizeof(float));

    // Generate n random floats on device
    // CURAND_CALL();
    // generates n vals between [0, 1]
    curandGenerateUniform(gen, devPRNVals, n);

    // uses the random numbers in a kernel and simply converts
    // them to a [-1, 1] space
    genPRNOnGPU(n, devPRNVals, devResults);

    /* Copy device memory to host */
    // CUDA_CALL(cudaMemcpy(hostData, devPRNVals, n * sizeof(float),
    // cudaMemcpyDeviceToHost));
    // CUDA_CALL();
    cudaMemcpy(hostResults, devResults, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // random numbers are generated between -1 and 1.  Avg should
    // be close to 0.0
    float avgVal = 0.0;
    for(i = 0; i < n; i++) {
      avgVal += hostResults[i];
    }
    avgVal /= float(n);

    float eps = 1.0e-1;
    std::cout << "avgVal = " << avgVal << std::endl;

    numParticles *= 2;

    // CUDA_CALL();
    cudaFree(devPRNVals);
        
    // CUDA_CALL();
    cudaFree(devResults);
        
    free(hostResults);
  }

  // Cleanup
  // CURAND_CALL();
  curandDestroyGenerator(gen);
}
