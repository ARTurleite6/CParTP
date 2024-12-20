#include "fluid_solver.h"
#include <algorithm>
#include <math.h>
#include <omp.h>

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

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
#pragma omp parallel for schedule(static) firstprivate(size)
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

#pragma omp parallel
  {
    auto neg_mask = (b == 3) ? -1.0F : 1.0F;
#pragma omp for collapse(2) schedule(static)
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
        x[IX(i, j, 0)] = neg_mask * x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = neg_mask * x[IX(i, j, O)];
      }
    }

    neg_mask = (b == 1) ? -1.0F : 1.0F;
#pragma omp for collapse(2) schedule(static)
    for (j = 1; j <= O; j++) {
      for (i = 1; i <= N; i++) {
        x[IX(0, i, j)] = neg_mask * x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = neg_mask * x[IX(M, i, j)];
      }
    }

    neg_mask = (b == 2) ? -1.0F : 1.0F;
#pragma omp for collapse(2) schedule(static) nowait
    for (j = 1; j <= O; j++) {
      for (i = 1; i <= M; i++) {
        x[IX(i, 0, j)] = neg_mask * x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = neg_mask * x[IX(i, N, j)];
      }
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

#if 1
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a,
               float c) {
  float tol = 1e-7, max_c;
  int l = 0;

  do {
    max_c = 0.0f;
#pragma omp parallel firstprivate(tol)
    {
#pragma omp for collapse(2) schedule(static) reduction(max : max_c)
      for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
          for (int i = 1 + ((j + k) & 1); i <= M; i += 2) {
            float old_x = x[IX(i, j, k)];
            float new_value = (x0[IX(i, j, k)] +
                               a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                    x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                    x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                              c;
            float change = fabs(new_value - old_x);
            max_c = std::fmax(max_c, change);

            x[IX(i, j, k)] = new_value;
          }
        }
      }

#pragma omp for collapse(2) schedule(static) reduction(max : max_c) nowait
      for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
          for (int i = 1 + ((j + k + 1) & 1); i <= M; i += 2) {
            float old_x = x[IX(i, j, k)];
            float new_value = (x0[IX(i, j, k)] +
                               a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                    x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                    x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                              c;
            float change = fabs(new_value - old_x);
            max_c = std::fmax(max_c, change);

            x[IX(i, j, k)] = new_value;
          }
        }
      }
    }
    set_bnd(M, N, O, b, x);
  } while (max_c > tol && ++l < 20);
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
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
#pragma omp parallel
  {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
#pragma omp for collapse(3) schedule(guided)
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
  }
  set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {
  const auto scale = -0.5f;

#pragma omp parallel for collapse(3) schedule(static) firstprivate(scale)
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

#pragma omp parallel for collapse(3) schedule(static) firstprivate(scale)
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
