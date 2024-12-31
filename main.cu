#include "EventManager.h"
#include "fluid_solver.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <optional>
#include <vector>

constexpr int SIZE{168};
#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

struct ChangeUVWValues {
  int u, v, w;
};

// Globals for the grid size
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

static int M = SIZE;
static int N = SIZE;
static int O = SIZE;

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;
static float *d_dens;

// helpers
bool *d_converged;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);

  dens = new float[size];

  size *= sizeof(float);

  cudaMalloc(&d_dens, size);
  cudaMalloc(&dens_prev, size);
  cudaMalloc(&u, size);
  cudaMalloc(&u_prev, size);
  cudaMalloc(&v, size);
  cudaMalloc(&v_prev, size);
  cudaMalloc(&w, size);
  cudaMalloc(&w_prev, size);

  cudaMalloc(&d_converged, sizeof(bool));

  if (!dens) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  cudaMemset(d_dens, 0, size);
  cudaMemset(dens_prev, 0, size);
  cudaMemset(u, 0, size);
  cudaMemset(u_prev, 0, size);
  cudaMemset(v, 0, size);
  cudaMemset(v_prev, 0, size);
  cudaMemset(w, 0, size);
  cudaMemset(w_prev, 0, size);
}

// Free allocated memory
void free_data() {
  cudaFree(d_dens);
  cudaFree(dens_prev);
  cudaFree(u);
  cudaFree(u_prev);
  cudaFree(v);
  cudaFree(v_prev);
  cudaFree(w);
  cudaFree(w_prev);
  cudaFree(d_converged);
  delete[] dens;
}

__global__ void update_dens(int index, float *dens_dev, float value) {
  dens_dev[index] = value;
}

__global__ void update_uvw(int index, float *u_dev, float *v_dev, float *w_dev,
                           float u_value, float v_value, float w_value) {
  u_dev[index] = u_value;
  v_dev[index] = v_value;
  w_dev[index] = w_value;
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
  int i = M / 2, j = N / 2, k = O / 2;
  const auto index = IX(i, j, k);
  std::optional<float> change_dens = std::nullopt;
  std::optional<ChangeUVWValues> change_uvw = std::nullopt;
  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      change_dens = event.density;
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      change_uvw = ChangeUVWValues{
          .u = event.force.x, .v = event.force.y, .w = event.force.z};
    }
  }

  if (change_dens.has_value()) {
    update_dens<<<1, 1>>>(index, d_dens, change_dens.value());
  }

  if (change_uvw.has_value()) {
    const auto value = change_uvw.value();
    update_uvw<<<1, 1>>>(index, u, v, w, value.u, value.v, value.w);
  }
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    apply_events(events);

    // Perform the simulation steps
    vel_step_cuda(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step_cuda(M, N, O, d_dens, dens_prev, u, v, w, diff, dt);
  }
  int size = (M + 2) * (N + 2) * (O + 2);
  cudaMemcpy(dens, d_dens, sizeof(float) * size, cudaMemcpyDeviceToHost);
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cuda Error: %s\n", cudaGetErrorString(err));
  }

  return 0;
}
