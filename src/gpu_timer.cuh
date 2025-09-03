#pragma once
#include <cuda.h>

/*
Simple timer class to time GPU functions. Remember kernels return immediately
and run asynchronously. This class will ensure proper sync/timing.
*/

struct timer {
  cudaEvent_t _start, _stop;

  timer() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  void start() {
    cudaEventRecord(_start);
  }

  float stop() {
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);
    return milliseconds;
  }
};
