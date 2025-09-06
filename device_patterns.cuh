#include "device_tensor.cuh"

/*
FILL KERNEL
This kernel will fill an allocatoin with a constant value 'val'.
*/

template <int N_DIMS>
__global__ void kernel_fill_apply(device_tensor<N_DIMS> x, const float val) {
  size_t i = threadIdx.x;
  while (i < x.get_n_elems()) {
    x.at_linear(i) = val;
    i += blockDim.x;
  }
}

// GPU kernel wrapper for fill_apply.
template <int N_DIMS>
void fill_apply(device_tensor<N_DIMS>& x, const float val) {
  kernel_fill_apply<N_DIMS><<<1, 32>>>(std::move(x), val);
}

/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and y and apply an elementwise operation
between them. i.e. out[i][j] = op::op(x[i][j], y[i][j])
*/
template <typename op, int N_DIMS>
__global__ void kernel_pointwise_apply(
    device_tensor<N_DIMS> out,
    const device_tensor<N_DIMS> x,
    const device_tensor<N_DIMS> y) {
  size_t i = threadIdx.x;
  while (i < out.get_n_elems()) {
    out.at_linear(i) = op::op(x.at_linear(i), y.at_linear(i));
    i += blockDim.x;
  }
}

// GPU kernel wrapper for pointwise apply
template <typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(
    device_tensor<N_DIMS> x,
    device_tensor<N_DIMS> y) {
  assert(x.get_n_elems() == y.get_n_elems());
  device_tensor<N_DIMS> out(x.size);
  kernel_pointwise_apply<op, N_DIMS><<<1, 32>>>(out, std::move(x), std::move(y));
  return out;
}

/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and y and apply an elementwise operation
between them. i.e. out[i][j] = op::op(x[i][j], y[i][j])
*/
template <typename op, int N_DIMS>
__global__ void kernel_pointwise_apply(
    device_tensor<N_DIMS> out,
    const device_tensor<N_DIMS> x,
    float y) {
  size_t i = threadIdx.x;
  while (i < out.get_n_elems()) {
    out.at_linear(i) = op::op(x.at_linear(i), y);
    i += blockDim.x;
  }
}

// GPU kernel wrapper for pointwise apply
template <typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(
    device_tensor<N_DIMS> x,
    float y) {
  device_tensor<N_DIMS> out(x.size);
  kernel_pointwise_apply<op, N_DIMS><<<1, 32>>>(out, std::move(x), std::move(y));
  return out;
}

/*
PPOINTWISE KERNEL
Will loop over elements of tensor x and apply an elementwise operation on it.
i.e. out[i][j] = op::op(x[i][j])
*/
template <typename op, int N_DIMS>
__global__ void kernel_pointwise_apply(
    device_tensor<N_DIMS> out,
    const device_tensor<N_DIMS> x) {
  size_t i = threadIdx.x;
  while (i < out.get_n_elems()) {
    out.at_linear(i) = op::op(x.at_linear(i));
    i += blockDim.x;
  }
}

// GPU kernel wrapper for pointwise apply
template <typename op, int N_DIMS>
device_tensor<N_DIMS> pointwise_apply(device_tensor<N_DIMS> x) {
  device_tensor<N_DIMS> out(x.size);
  kernel_pointwise_apply<op, N_DIMS><<<1, 32>>>(out, std::move(x));
  return out;
}

/* REDUCTION KERNEL
Takes a 2D tensor and reduces its second dimension based on op::op.
i.e. out[i] = init_value
     for j in J:
         out[i] = op::op(out[i], A[i, j])
*/
template <typename op>
__global__ void reduce_dim_1(device_tensor<1> out, const device_tensor<2> in) {
  size_t i = threadIdx.x;
  while (i < in.size[0]) {
    float red = op::init();
    for (size_t j = 0; j < in.size[1]; j++) {
      red = op::op(in.at(i, j), red);
    }
    out.at(i) = red;
    i += blockDim.x;
  }
}

// GPU kernel wrapper for reduce_dim_1
template <typename op>
device_tensor<1> reduce_apply(device_tensor<2> x) {
  device_tensor<1> out({x.size[0]});
  reduce_dim_1<op><<<1, 32>>>(out, std::move(x));
  return out;
}

/* BROADCAST KERNELS */
// Broadcasts tensor and applies to another in element wise fasion.
// i.e. out [i, j] = op::op(A[i] + B[i, j])
template <typename op>
__global__ void kernel_broadcast_apply(
    device_tensor<2> out,
    const device_tensor<1> x,
    const device_tensor<2> y) {
  size_t i = threadIdx.x;
  while (i < x.size[0]) {
    for (size_t j = 0; j < y.size[1]; j++) {
      out.at(i, j) = op::op(x.at(i), y.at(i, j));
    }
    i += blockDim.x;
  }
}

/* BROADCAST KERNELS */
// Broadcasts tensor and applies to another in element wise fasion.
// i.e. out [i, j] = op::op(A[i, j] + B[i])
template <typename op>
__global__ void kernel_broadcast_apply(
    device_tensor<2> out,
    const device_tensor<2> x,
    const device_tensor<1> y) {
  size_t i = threadIdx.x;
  while (i < x.size[0]) {
    for (size_t j = 0; j < x.size[1]; j++) {
      out.at(i, j) = op::op(x.at(i, j), y.at(i));
    }
    i += blockDim.x;
  }
}

// GPU kernel wrapper for first broadcast kernel
template <typename op>
device_tensor<2> broadcast_apply(
    device_tensor<2> x,
    device_tensor<1> y) {
  assert(x.size[0] == y.get_n_elems());
  device_tensor<2> out(x.size);
  kernel_broadcast_apply<op><<<1, 32>>>(out, std::move(x), std::move(y));
  return out;
}

// GPU kernel wrapper for second broadcast kernel
template <typename op>
device_tensor<2> broadcast_apply(
    device_tensor<1> x,
    device_tensor<2> y) {
  assert(x.get_n_elems() == y.size[0]);
  device_tensor<2> out(y.size);
  kernel_broadcast_apply<op><<<1, 32>>>(out, std::move(x), std::move(y));
  return out;
}
