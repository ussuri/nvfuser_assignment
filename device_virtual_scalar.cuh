/*
 * A virtual scalar tensor class representing an infinitely extended constant
 * view while using minimal RAM.
 */

#pragma once

#include <cuda_device_runtime_api.h>
#include "tensor.cuh"

template <int N_DIMS>
class device_virtual_scalar : public tensor<N_DIMS> {
  struct view {
    explicit view(const device_virtual_scalar<N_DIMS>& parent)
        : parent{parent} {}
    template <typename T>
    view& operator+(const view& v, const T& t) const {
      return *this;
    }
    view& operator*() const {
      return parent.scalar_;
    }
  };

  void alloc_data() override {}

  __host__ __device__ float* get() const override {
    return allocation;
  }

  float scalar_ = 0;

 public:
  // Allocate device tensor
  device_virtual_scalar(const std::array<size_t, N_DIMS>& size, float scalar)
      : tensor<N_DIMS>(size), scalar_{scalar} {}

  void copy(const device_tensor<N_DIMS>&) override {
    __builtin_unreachable();
  }
  void copy(const host_tensor<N_DIMS>&) override {
    __builtin_unreachable();
  }
  void fill_random() override {
    __builtin_unreachable();
  }
};
