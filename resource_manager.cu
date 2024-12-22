#include "resource_manager.h"
#include <cuda_runtime.h>

ResourceManager::ResourceManager() : size((M + 2) * (N + 2) * (O + 2)) {
  cudaMalloc(&x_dev, size * sizeof(float));
  cudaMalloc(&x0_dev, size * sizeof(float));
  cudaMalloc(&d_dev, size * sizeof(float));
  cudaMalloc(&d0_dev, size * sizeof(float));
  cudaMalloc(&u_dev, size * sizeof(float));
  cudaMalloc(&v_dev, size * sizeof(float));
  cudaMalloc(&w_dev, size * sizeof(float));
}

ResourceManager::~ResourceManager() {
  cudaFree(x_dev);
  cudaFree(x0_dev);
}
