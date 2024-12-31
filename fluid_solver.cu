#include "fluid_solver.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define BLOCK_SIZE 4
#define ELEMENTS_PER_THREAD 32

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
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

void set_bnd_cuda(int M, int N, int O, int b, float *x_dev) {
  // Set up dimensions for face kernels
  dim3 blockDim(256);

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

template <unsigned int blockSize>
__global__ void reduce_max(float *input, float *output, int n) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * ELEMENTS_PER_THREAD) + tid;

  float thread_max = -FLT_MAX;
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    if (idx + i * blockDim.x < n) {
      thread_max = max(thread_max, input[idx + i * blockDim.x]);
    }
  }
  sdata[tid] = thread_max;
  __syncthreads();

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] = max(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] = max(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile float *smem = sdata;
    if (blockSize >= 64)
      smem[tid] = max(smem[tid], smem[tid + 32]);
    if (blockSize >= 32)
      smem[tid] = max(smem[tid], smem[tid + 16]);
    if (blockSize >= 16)
      smem[tid] = max(smem[tid], smem[tid + 8]);
    if (blockSize >= 8)
      smem[tid] = max(smem[tid], smem[tid + 4]);
    if (blockSize >= 4)
      smem[tid] = max(smem[tid], smem[tid + 2]);
    if (blockSize >= 2)
      smem[tid] = max(smem[tid], smem[tid + 1]);
  }

  if (tid == 0)
    output[blockIdx.x] = sdata[0];
}

float find_max(float *d_input, float *partialMax, int size) {
  constexpr int threadsPerBlock = 256;
  constexpr int elementsPerThread = ELEMENTS_PER_THREAD;
  int numBlocks = (size + (threadsPerBlock * elementsPerThread) - 1) /
                  (threadsPerBlock * elementsPerThread);

  if (numBlocks > 1) {
    reduce_max<threadsPerBlock>
        <<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_input, partialMax, size);
    reduce_max<threadsPerBlock>
        <<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            partialMax, partialMax, numBlocks);
  } else {
    reduce_max<threadsPerBlock>
        <<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            d_input, partialMax, size);
  }

  float h_max;
  cudaMemcpy(&h_max, partialMax, sizeof(float), cudaMemcpyDeviceToHost);
  return h_max;
}

__global__ void lin_solve_kernel_with_shared(int M, int N, int O, float *x,
                                             const float *x0, float a, float c,
                                             int color, float *changes) {
  extern __shared__ float s_x[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // in here we need to subtract 2 in each blockDim so we can overlap the
  // borders, for example Block 1: 0-15, Block 2: 13-29
  // this is because we need to store the borders too in the shared_memory to
  // make the computation
  int i = blockIdx.x * (blockDim.x - 2) + tx;
  int j = blockIdx.y * (blockDim.y - 2) + ty;
  int k = blockIdx.z * (blockDim.z - 2) + tz;

  int x_shared_mem_size = blockDim.x + 2;
  int y_shared_mem_size = blockDim.y + 2;

  int tid = (tz * y_shared_mem_size + ty) * x_shared_mem_size + tx;

  if (i <= M + 1 && j <= N + 1 && k <= O + 1) {
    s_x[tid] = x[IX(i, j, k)];
  }

  __syncthreads();

  if (tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1 &&
      tz > 0 && tz < blockDim.z - 1 && i <= M && j <= N && k <= O &&
      (i + j + k) % 2 == color) {
    int idx = IX(i, j, k);
    float old_x = s_x[tid];
    // in this we use the x_shared_mem_size and such to get the element (y + 1)
    // and so on
    float new_x =
        (x0[idx] +
         a * (s_x[tid - 1] + s_x[tid + 1] + s_x[tid - x_shared_mem_size] +
              s_x[tid + x_shared_mem_size] +
              s_x[tid - x_shared_mem_size * y_shared_mem_size] +
              s_x[tid + x_shared_mem_size * y_shared_mem_size])) /
        c;

    x[idx] = new_x;
    // here we store all the changes so we can make the reduction later
    changes[idx - 1] = fabs(new_x - old_x);
  }
}

__global__ void lin_solve_kernel(int M, int N, int O, float *x, const float *x0,
                                 float a, float c, int color, bool *converged) {
  float tol = 1e-7;
  int j = (blockIdx.y * blockDim.y + threadIdx.y) + 1;
  int k = (blockIdx.z * blockDim.z + threadIdx.z) + 1;
  int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1 + (j + k + color) % 2;

  // float local_max = 0.0f;

  if (i <= M && j <= N && k <= O) {
    // Check if this cell matches the current color pattern
    int idx = IX(i, j, k);
    float old_x = x[idx];
    x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
             c;
    if (fabs(x[idx] - old_x) > tol) {
      *converged = false;
    }
    // changes[idx - 1] = fabs(x[idx] - old_x);
  }
}

// Modify your lin_solve_cuda function
void lin_solve_cuda(int M, int N, int O, int b, float *x, float *x0, float a,
                    float c) {
  const int max_iterations = 20;

  dim3 blockDim(32, 8, 1);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (O + blockDim.z - 1) / blockDim.z);

  bool converged = true;
  extern bool *d_converged;

  int iteration = 0;
  do {
    cudaMemset(d_converged, true, sizeof(bool));
    lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, 0,
                                            d_converged);
    lin_solve_kernel<<<gridDim, blockDim>>>(M, N, O, x, x0, a, c, 1,
                                            d_converged);

    set_bnd_cuda(M, N, O, b, x);

    cudaMemcpy(&converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

  } while (!converged && ++iteration < max_iterations);
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
  dim3 blockDim(32, 8, 1);
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
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
  dim3 blockDim(32, 8, 1);
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
