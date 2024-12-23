#include "resource_manager.h"
#include <cuda_runtime.h>

ResourceManager::ResourceManager() : size((M + 2) * (N + 2) * (O + 2)) {
  cudaMalloc(&d_dev, size * sizeof(float));
  cudaMalloc(&d0_dev, size * sizeof(float));
  cudaMalloc(&u_dev, size * sizeof(float));
  cudaMalloc(&u0_dev, size * sizeof(float));
  cudaMalloc(&v_dev, size * sizeof(float));
  cudaMalloc(&v0_dev, size * sizeof(float));
  cudaMalloc(&w_dev, size * sizeof(float));
  cudaMalloc(&w0_dev, size * sizeof(float));
  cudaMalloc(&max_change, sizeof(float));

  cudaMemset(d_dev, 0, size * sizeof(float));
  cudaMemset(d0_dev, 0, size * sizeof(float));
  cudaMemset(u_dev, 0, size * sizeof(float));
  cudaMemset(u0_dev, 0, size * sizeof(float));
  cudaMemset(v_dev, 0, size * sizeof(float));
  cudaMemset(v0_dev, 0, size * sizeof(float));
  cudaMemset(w_dev, 0, size * sizeof(float));
  cudaMemset(w0_dev, 0, size * sizeof(float));
}

ResourceManager::~ResourceManager() {
  cudaFree(d_dev);
  cudaFree(d0_dev);
  cudaFree(u_dev);
  cudaFree(u0_dev);
  cudaFree(v_dev);
  cudaFree(v0_dev);
  cudaFree(w_dev);
  cudaFree(w0_dev);
  cudaFree(max_change);
}
