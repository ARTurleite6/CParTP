#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <iostream>
constexpr int SIZE{84};

#include <cstddef>

class ResourceManager {
public:
  float *x_dev, *x0_dev;
  float *d_dev, *d0_dev, *u_dev, *v_dev, *w_dev;

public:
  static ResourceManager &getInstance() {
  static ResourceManager instance;
    return instance;
  }

  ~ResourceManager();

  inline int getSize() const noexcept { return size; }

  static constexpr int M = SIZE;
  static constexpr int N = SIZE;
  static constexpr int O = SIZE;

  // Prevent copying
  ResourceManager(const ResourceManager &) = delete;
  ResourceManager &operator=(const ResourceManager &) = delete;

private:
  ResourceManager();

  size_t size;
};

#endif // RESOURCE_MANAGER_H
