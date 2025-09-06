/*
This file holds the instace of the device_tensor. This tensor can be constructed
from host_tensors and in doing so data will automatically be moved over to the
GPU. Be careful of the copy constructor. This copy constructor is useful for
shipping the class to the GPU as a kernel arguemnt, however, using it directly
will make a shallow copy of this object and will share the same GPU memory
buffer as the other tensor.

It is unlikely you need to modify this file, but for the brave...
*/

#pragma once

#include <cuda_device_runtime_api.h>
#include "host_tensor.cuh"
#include "tensor.cuh"

template <int N_DIMS>
class device_tensor : public tensor<N_DIMS> {
  friend host_tensor<N_DIMS>;

  // Allocate data
  void alloc_data() override {
    cudaMalloc(&(this->allocation), this->get_n_elems() * sizeof(float));
    /*
    Pass a custom deleter for the shared_ptr.
    shared_ptr is only used to reference count the allocations shared if the
    copy constructor is used. This ptr can not be used on device, as CUDA will
    dumbly copy the memory used for it over, but functions of the shared_ptr
    are not usable on device.
    */
    this->data = std::shared_ptr<float>(
        this->allocation, [](float* p) { cudaFree(p); });
  }

 public:
  // Allocate device tensor
  device_tensor(std::array<size_t, N_DIMS> size, bool rand = false)
      : tensor<N_DIMS>(std::move(size)) {
    alloc_data();
    if (rand)
      fill_random();
  }

  // Copy data of one tensor over the buffer of another.
  device_tensor<N_DIMS>& operator=(const device_tensor<N_DIMS>& other) {
    this->copy(other);
    return *this;
  }

  // Copy constructor, useful to send data to device, careful as multiple
  // device_tensors can point to the same data buffer if this is used.
  device_tensor(const device_tensor<N_DIMS>& other)
      : tensor<N_DIMS>(other.size) {
    this->data = other.data;
    this->allocation = this->data.get();
  }

  // Make a new tensor from a device tensor, allocates new data
  device_tensor(const device_tensor<N_DIMS>& other, bool copy_data)
      : tensor<N_DIMS>(other.size) {
    alloc_data();
    if (copy_data)
      this->copy(other);
  }

  // Make a new tensor from a host tensor, allocates data copies the data from
  // the host tensor
  device_tensor(const host_tensor<N_DIMS>& other, bool copy_data = true)
      : tensor<N_DIMS>(other.size) {
    alloc_data();
    if (copy_data)
      this->copy(other);
  }

  void copy(const device_tensor<N_DIMS>&) override;
  void copy(const host_tensor<N_DIMS>&) override;
  void fill_random() override;
};

using device_scalar = device_tensor<0>;
using device_vector = device_tensor<1>;
using device_matrix = device_tensor<2>;