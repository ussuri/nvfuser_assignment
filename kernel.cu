#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_patterns.cuh"
#include "device_tensor.cuh"
#include "gpu_timer.cuh"
#include "host_tensor.cuh"
#include "ops.cuh"
#include "utils.cuh"

// Reference CPU implementation
// Do not change this function, It is the reference implementation for you to
// match!
host_tensor<2> op_and_normalize(host_tensor<2>& input) {
  for (int i = 0; i < input.get_n_elems(); i++) {
    float val = input.at_linear(i);
    input.at_linear(i) = sinh((double)val / 1.9);
  }

  host_tensor<1> ave({input.size[0]});
  for (int i = 0; i < input.size[0]; i++) {
    float summ = 0.0;
    for (int j = 0; j < input.size[1]; j++) {
      summ += input.at(i, j);
    }
    ave.at(i) = summ / float(input.size[1]);
  }

  host_tensor<1> std_dev_sq({input.size[0]});
  for (int i = 0; i < input.size[0]; i++) {
    float summ = 0.0;
    for (int j = 0; j < input.size[1]; j++) {
      float diff = input.at(i, j) - ave.at(i);
      summ += diff * diff;
    }
    std_dev_sq.at(i) = summ / float(input.size[1]);
  }

  host_tensor<2> out(input.size);
  for (int i = 0; i < input.size[0]; i++) {
    for (int j = 0; j < input.size[1]; j++) {
      out.at(i, j) =
          (input.at(i, j) - ave.at(i)) / sqrtf(std_dev_sq.at(i) + 1e-14);
    }
  }

  return out;
}

// GPU implementation
// This is a sample GPU implementation, anything and nothing can be kept from it

// NOTE: Must be global to be used as a template param.
namespace {
constexpr float epsilon = 1e-14;
} // namespace

device_tensor<2> op_and_normalize(device_tensor<2>& input) {
#if 1
  device_tensor<2> scale(input, false);
  fill_apply<2>(scale, 1.9);
  input = pointwise_apply<div_op<>, 2>(input, scale);
  input = pointwise_apply<sinh_op<>, 2>(input);

  device_tensor<1> ave = reduce_apply<add_op<>>(input);

  device_tensor<1> n(ave, false);
  fill_apply<1>(n, (float)input.size[1]);

  ave = pointwise_apply<div_op<>, 1>(ave, n);

  device_tensor<2> diff = broadcast_apply<sub_op>(input, ave);
  device_tensor<2> diff_sq = pointwise_apply<square_op<>>(diff);

  device_tensor<1> std_dev_sq = reduce_apply<add_op<>>(diff_sq);
  std_dev_sq = pointwise_apply<div_op<>>(std_dev_sq, n);
  device_tensor<1> epsilon(std_dev_sq, false);
  fill_apply<1>(epsilon, 1e-14);
  std_dev_sq = pointwise_apply<add_op<>>(std_dev_sq, epsilon);

  device_tensor<2> inp_m_ave = broadcast_apply<sub_op>(input, ave);
  device_tensor<1> std_dev = pointwise_apply<square_root_op<>>(std_dev_sq);

  return broadcast_apply<div_op<>>(inp_m_ave, std_dev);
#else
  device_tensor<1> n({input.size[0]});
  fill_apply<1>(n, (float)input.size[1]);
    
  input = pointwise_apply<add_op<sinh_op<scale_op<19, 10>>>>(input);

  device_tensor<1> ave = reduce_apply<div_op<>>(input, n);

  device_tensor<2> diff_sq = broadcast_apply<square_op<sub_op>>(input, ave);
  device_tensor<1> std_dev_sq = reduce_apply<add_op<>>(diff_sq);
  std_dev_sq = pointwise_apply<div_op<>>(std_dev_sq, n);
  device_tensor<1> std_dev = pointwise_apply<square_root_op<>>(std_dev_sq);

  device_tensor<2> inp_m_ave = broadcast_apply<sub_op>(input, ave);

  device_tensor<2> res = pointwise_apply<div_op<>>(inp_m_ave, std_dev);

  return res;
#endif
}

// Compares a host tensor and device tensor and returns mas abs difference
// between them
template <int N_DIMS>
float check_result(
    const host_tensor<N_DIMS>& A,
    const device_tensor<N_DIMS>& C) {
  host_tensor<N_DIMS> B(C, true);
  assert(A.get_n_elems() == B.get_n_elems());
  float max_diff = 0.0;
  for (int i = 0; i < A.get_n_elems(); i++) {
    max_diff = std::max(max_diff, abs(A.at_linear(i) - B.at_linear(i)));
  }
  return max_diff;
}

// Size to run
constexpr uint32_t M = 1024 * 4;
constexpr uint32_t N = 1024;
constexpr uint32_t ITERATIONS = 8;

#define VV(x) #x "=" << (x) << " "

int main() {
  /*
     Do not change this section of code, this is how the user expects to
     interact with your implementation. hA and hOut is the reference
     implementation. hA data will be copied to dA so the input to the GPU
     function will match that of the reference. This is the tensor the user is
     expecting to give to your implementation and dOut is the tensor the user is
     expecting back from your implementation.
  */
  // Input tensor
  host_tensor<2> hA({M, N}, true);
  host_tensor<2> hOut(hA, true);

  // Make copy for device ops, need to grab random numbers in hA.
  device_tensor<2> dA(hA);
  device_tensor<2> dOut(dA, true);

  // Run the CPU ops ITERATION times sequentially.
  for (int i = 0; i < ITERATIONS; i++) {
    hOut = op_and_normalize(hOut);
  }

  // Run the GPU ops ITERATIONS times sequentially
  // As long as dOut matches hOut you can modify anything
  // that is executed in between t.start() and t.stop().
  timer t;
  t.start();
  for (int i = 0; i < ITERATIONS; i++) {
    dOut = op_and_normalize(dOut);
  }
  float ms = t.stop();

  // Print the amount of time required by the gpu implementation.
  std::cout << "Finished in " << ms << " ms." << std::endl;
  if constexpr (false) {
    const host_tensor<2> dOutCopy{dOut};
    std::cout << VV(hOut) << std::endl << VV(dOutCopy) << std::endl;
  }

  // Make sure the result of your implementation is correct.
  const auto maxDiff = check_result(hOut, dOut);
  std::cout << VV(maxDiff);
  if (maxDiff > 1e-4) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
