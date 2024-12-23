#include "EventManager.h"
#include "fluid_solver.h"
#include "resource_manager.h"
#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

static int M = ResourceManager::M;
static int N = ResourceManager::N;
static int O = ResourceManager::O;

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;
}

__global__ void update_dens(int index, float *dens_dev, float value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dens_dev[index] = value;
  }
}

__global__ void update_uvw(int index, float *u_dev, float *v_dev, float *w_dev,
                           float u_value, float v_value, float w_value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    u_dev[index] = u_value;
    v_dev[index] = v_value;
    w_dev[index] = w_value;
  }
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events, ResourceManager &rm) {
  int i = M / 2, j = N / 2, k = O / 2;
  const auto index = IX(i, j, k);
  bool changed_dens = false;
  bool changed_uvw = false;
  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      dens[index] = event.density;
      changed_dens = true;
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      u[index] = event.force.x;
      v[index] = event.force.y;
      w[index] = event.force.z;
      changed_uvw = true;
    }
  }

  if (changed_dens) {
    update_dens<<<1, 1>>>(index, rm.d_dev, dens[index]);
  }

  if (changed_uvw) {
    update_uvw<<<1, 1>>>(index, rm.u_dev, rm.v_dev, rm.w_dev, u[index],
                         v[index], w[index]);
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
  ResourceManager &instance = ResourceManager::getInstance();

  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    apply_events(events, instance);

    // Perform the simulation steps
    vel_step_cuda(M, N, O, instance.u_dev, instance.v_dev, instance.w_dev,
                  instance.u0_dev, instance.v0_dev, instance.w0_dev, visc, dt);
    dens_step_cuda(M, N, O, instance.d_dev, instance.d0_dev, instance.u_dev,
                   instance.v_dev, instance.w_dev, diff, dt);
  }
  cudaMemcpy(dens, instance.d_dev, sizeof(float) * instance.getSize(),
             cudaMemcpyDeviceToHost);
}

int main() {
  cudaSetDevice(0);

  cudaProfilerStart();
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

  cudaProfilerStop();

  return 0;
}
