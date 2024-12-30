#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <iostream>
constexpr int SIZE{164};

#include <cstddef>

class ResourceManager {
public:
  float *d_dev, *d0_dev, *u_dev, *u0_dev, *v_dev, *v0_dev, *w_dev, *w0_dev;
  float *max_change;
  float *changes;
  float *partialMax;

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
