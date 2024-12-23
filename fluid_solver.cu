#include "fluid_solver.h"
#include "resource_manager.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 4

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

__device__ float clamp(float val, float minVal, float maxVal) {
  return max(min(val, maxVal), minVal);
}

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s,
                                  float dt) {
  // Calculate the 1D index of the current thread
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of elements in the array
  int size = (M + 2) * (N + 2) * (O + 2);

  // Perform the operation if the thread index is within bounds
  if (idx < size) {
    x[idx] += dt * s[idx];
  }
}

// Add sources (density or velocity)
void add_source_cuda(int M, int N, int O, float *x_dev, float *s_dev,
                     float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);

  // Configure block and grid sizes
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel
  add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, x_dev, s_dev,
                                                        dt);
}

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  auto neg_mask = (b == 3) ? -1.0F : 1.0F;
  for (int j = 1; j <= N; j++) {
    for (int i = 1; i <= M; i++) {
      x[IX(i, j, 0)] = neg_mask * x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = neg_mask * x[IX(i, j, O)];
    }
  }

  neg_mask = (b == 1) ? -1.0F : 1.0F;
  for (j = 1; j <= O; j++) {
    for (i = 1; i <= N; i++) {
      x[IX(0, i, j)] = neg_mask * x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = neg_mask * x[IX(M, i, j)];
    }
  }

  neg_mask = (b == 2) ? -1.0F : 1.0F;
  for (j = 1; j <= O; j++) {
    for (i = 1; i <= M; i++) {
      x[IX(i, 0, j)] = neg_mask * x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = neg_mask * x[IX(i, N, j)];
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

// Kernel for setting z-faces (top and bottom boundaries)
__global__ void set_bnd_z_faces(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= M && j <= N) {
    float neg_mask = (b == 3) ? -1.0f : 1.0f;
    x[IX(i, j, 0)] = neg_mask * x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = neg_mask * x[IX(i, j, O)];
  }
}

// Kernel for setting x-faces (left and right boundaries)
__global__ void set_bnd_x_faces(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= N && j <= O) {
    float neg_mask = (b == 1) ? -1.0f : 1.0f;
    x[IX(0, i, j)] = neg_mask * x[IX(1, i, j)];
    x[IX(M + 1, i, j)] = neg_mask * x[IX(M, i, j)];
  }
}

// Kernel for setting y-faces (front and back boundaries)
__global__ void set_bnd_y_faces(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (i <= M && j <= O) {
    float neg_mask = (b == 2) ? -1.0f : 1.0f;
    x[IX(i, 0, j)] = neg_mask * x[IX(i, 1, j)];
    x[IX(i, N + 1, j)] = neg_mask * x[IX(i, N, j)];
  }
}

// Kernel for setting corners
__global__ void set_bnd_corners(int M, int N, int O, float *x) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

    x[IX(M + 1, 0, 0)] =
        0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

    x[IX(0, N + 1, 0)] =
        0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                      x[IX(M + 1, N + 1, 1)]);
  }
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    // Set z-faces (top and bottom boundaries)
    if (k == 0) {
      float neg_mask = (b == 3) ? -1.0f : 1.0f;
      x[IX(i, j, 0)] = neg_mask * x[IX(i, j, 1)];
    } else if (k == O + 1) {
      float neg_mask = (b == 3) ? -1.0f : 1.0f;
      x[IX(i, j, O + 1)] = neg_mask * x[IX(i, j, O)];
    }
    // Set x-faces (left and right boundaries)
    else if (i == 0) {
      float neg_mask = (b == 1) ? -1.0f : 1.0f;
      x[IX(0, j, k)] = neg_mask * x[IX(1, j, k)];
    } else if (i == M + 1) {
      float neg_mask = (b == 1) ? -1.0f : 1.0f;
      x[IX(M + 1, j, k)] = neg_mask * x[IX(M, j, k)];
    }
    // Set y-faces (front and back boundaries)
    else if (j == 0) {
      float neg_mask = (b == 2) ? -1.0f : 1.0f;
      x[IX(i, 0, k)] = neg_mask * x[IX(i, 1, k)];
    } else if (j == N + 1) {
      float neg_mask = (b == 2) ? -1.0f : 1.0f;
      x[IX(i, N + 1, k)] = neg_mask * x[IX(i, N, k)];
    }
    // Set corners
    else if (i == 1 && j == 1 && k == 1) {
      x[IX(0, 0, 0)] =
          0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    } else if (i == M && j == 1 && k == 1) {
      x[IX(M + 1, 0, 0)] =
          0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    } else if (i == 1 && j == N && k == 1) {
      x[IX(0, N + 1, 0)] =
          0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    } else if (i == M && j == N && k == 1) {
      x[IX(M + 1, N + 1, 0)] =
          0.33f *
          (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
    }
  }
}

void set_bnd_cuda(int M, int N, int O, int b, float *x_dev) {
  // Set up dimensions for face kernels
  dim3 blockDim(16, 16);

  // For z-faces (M x N threads needed)
  dim3 gridDim_z((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
  set_bnd_z_faces<<<gridDim_z, blockDim>>>(M, N, O, b, x_dev);

  // For x-faces (N x O threads needed)
  dim3 gridDim_x((N + blockDim.x - 1) / blockDim.x,
                 (O + blockDim.y - 1) / blockDim.y);
  set_bnd_x_faces<<<gridDim_x, blockDim>>>(M, N, O, b, x_dev);

  // For y-faces (M x O threads needed)
  dim3 gridDim_y((M + blockDim.x - 1) / blockDim.x,
                 (O + blockDim.y - 1) / blockDim.y);
  set_bnd_y_faces<<<gridDim_y, blockDim>>>(M, N, O, b, x_dev);

  // For corners (single thread is enough)
  set_bnd_corners<<<1, 1>>>(M, N, O, x_dev);
}

// CUDA kernel for red-black Gauss-Seidel iteration
__global__ void lin_solve_kernel(int M, int N, int O, float *x, const float *x0,
                                 float a, float c, int color,
                                 float *max_change) {
  __shared__ float block_max[512];
  int tid = threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  float local_max = 0.0;

  if (i <= M && j <= N && k <= O) {
    // Check if this cell matches the current color pattern
    if (((i + j + k) & 1) == color) {
      float old_x = x[IX(i, j, k)];
      float new_value =
          (x0[IX(i, j, k)] +
           a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] +
                x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
          c;

      float change = fabsf(new_value - old_x);
      local_max = fmaxf(local_max, change);
      x[IX(i, j, k)] = new_value;
    }
  }

  block_max[tid] = local_max;
  __syncthreads();

  // Perform block-wise reduction in shared memory
  for (int stride = blockDim.x * blockDim.y * blockDim.z / 2; stride > 0;
       stride /= 2) {
    if (tid < stride) {
      block_max[tid] = fmaxf(block_max[tid], block_max[tid + stride]);
    }
    __syncthreads();
  }

  // Write block maximum to global memory
  if (tid == 0) {
    atomicMax((int *)max_change, __float_as_int(block_max[0]));
  }
}

__global__ void check_convergence_kernel(float *max_change, float tolerance,
                                         bool *converged) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *converged = (*max_change < tolerance);
  }
}

// Host function to manage the CUDA implementation
void lin_solve_cuda(int M, int N, int O, int b, float *x, float *x0, float a,
                    float c) {
  float tol = 1e-7;
  const int max_iterations = 20;
  bool *converged;
  bool converged_host = false;
  cudaMalloc(&converged, sizeof(bool));

  auto &instance = ResourceManager::getInstance();

  // Set up grid and block dimensions
  dim3 blockDim(8, 8, 8);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (O + blockDim.z - 1) / blockDim.z);

  int iteration = 0;
  do {
    // Reset max_change for this iteration
    cudaMemset(instance.max_change, 0, sizeof(float));

    // Red sweep (color = 0)
    lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, 0,
                                            instance.max_change);

    // Black sweep (color = 1)
    lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, 1,
                                            instance.max_change);

    check_convergence_kernel<<<1, 1>>>(instance.max_change, tol, converged);

    // Handle boundary conditions
    set_bnd_cuda(M, N, O, b, x);

    // Copy max_change back to host to check convergence
    cudaMemcpy(&converged_host, converged, sizeof(bool),
               cudaMemcpyDeviceToHost);

    iteration++;
  } while (!converged_host && iteration < max_iterations);
}

#if 0
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x,
               float *x0, float a, float c) {
 
}

#else

// Linear solve for implicit methods (diffusion)
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,
               float c) {
  const float a_div_c = a / c;
  const float inv_c = 1.0f / c;

  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int kk = 1; kk <= O; kk += BLOCK_SIZE) {
      for (int jj = 1; jj <= N; jj += BLOCK_SIZE) {
        for (int ii = 1; ii <= M; ii += BLOCK_SIZE) {
          for (int k = kk; k < kk + BLOCK_SIZE && k <= O; k++) {
            for (int j = jj; j < jj + BLOCK_SIZE && j <= N; j++) {
              for (int i = ii; i < ii + BLOCK_SIZE && i <= M; i++) {
                const auto index = IX(i, j, k);
                const auto result =
                    (x0[index] * inv_c +
                     a_div_c * (x[index - 1] + x[index + 1] +
                                x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)]));
                x[index] = result;
              }
            }
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  }
}

#endif

// Diffusion step (uses implicit method)
void diffuse_cuda(int M, int N, int O, int b, float *x, float *x0, float diff,
                  float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve_cuda(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0,
                              float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    const auto index = IX(i, j, k);
    float x = i - dtX * u[index];
    float y = j - dtY * v[index];
    float z = k - dtZ * w[index];

    x = clamp(x, 0.5f, M + 0.5f);
    y = clamp(y, 0.5f, N + 0.5f);
    z = clamp(z, 0.5f, O + 0.5f);

    int i0 = static_cast<int>(x), i1 = i0 + 1;
    int j0 = static_cast<int>(y), j1 = j0 + 1;
    int k0 = static_cast<int>(z), k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    d[index] = s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                     t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
               s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                     t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
  }
}

void advect_cuda(int M, int N, int O, int b, float *d, float *d0, float *u,
                 float *v, float *w, float dt) {
  // Set up grid and block dimensions
  dim3 blockDim(8, 8, 8);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (O + blockDim.z - 1) / blockDim.z);
  advect_kernel<<<gridDim, blockDim>>>(M, N, O, b, d, d0, u, v, w, dt);

  set_bnd_cuda(M, N, O, b, d);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        const auto index = IX(i, j, k);
        float x = i - dtX * u[index];
        float y = j - dtY * v[index];
        float z = k - dtZ * w[index];

        x = std::clamp(x, 0.5f, M + 0.5f);
        y = std::clamp(y, 0.5f, N + 0.5f);
        z = std::clamp(z, 0.5f, O + 0.5f);

        int i0 = static_cast<int>(x), i1 = i0 + 1;
        int j0 = static_cast<int>(y), j1 = j0 + 1;
        int k0 = static_cast<int>(z), k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[index] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
      }
    }
  }
  set_bnd(M, N, O, b, d);
}

__global__ void compute_divergence_kernel(int M, int N, int O, const float *u,
                                          const float *v, const float *w,
                                          float *div, float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
    int idx = IX(i, j, k);
    div[idx] = -0.5f *
               ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)])) /
               MAX(M, MAX(N, O));

    p[idx] = 0.0f; // Initialize pressure to zero
  }
}

__global__ void update_velocity_kernel(int M, int N, int O, float *u, float *v,
                                       float *w, const float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
    int idx = IX(i, j, k);
    u[idx] += -0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[idx] += -0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[idx] += -0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
  }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
  const auto scale = -0.5f;

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        div[IX(i, j, k)] =
            scale *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            MAX(M, MAX(N, O));
        p[IX(i, j, k)] = 0;
      }
    }
  }
  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        u[IX(i, j, k)] += scale * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] += scale * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] += scale * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project_cuda(int M, int N, int O, float *u, float *v, float *w, float *p,
                  float *div) {
  dim3 blockDim(8, 8, 8);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (O + blockDim.z) / blockDim.z);

  // Step 1: Compute divergence and initialize pressure to zero
  compute_divergence_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, div, p);

  set_bnd_cuda(M, N, O, 0, div);
  set_bnd_cuda(M, N, O, 0, p);
  lin_solve_cuda(M, N, O, 0, p, div, 1, 6);

  update_velocity_kernel<<<gridDim, blockDim>>>(M, N, O, u, v, w, p);

  set_bnd_cuda(M, N, O, 1, u);
  set_bnd_cuda(M, N, O, 2, v);
  set_bnd_cuda(M, N, O, 3, w);
}

void dens_step_cuda(int M, int N, int O, float *x, float *x0, float *u,
                    float *v, float *w, float diff, float dt) {
  add_source_cuda(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse_cuda(M, N, O, 0, x, x0, diff, dt);
  SWAP(x, x0);

  advect_cuda(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step_cuda(int M, int N, int O, float *u, float *v, float *w, float *u0,
                   float *v0, float *w0, float visc, float dt) {
  add_source_cuda(M, N, O, u, u0, dt);
  add_source_cuda(M, N, O, v, v0, dt);
  add_source_cuda(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse_cuda(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse_cuda(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse_cuda(M, N, O, 3, w, w0, visc, dt);
  project_cuda(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect_cuda(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect_cuda(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect_cuda(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project_cuda(M, N, O, u, v, w, u0, v0);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
