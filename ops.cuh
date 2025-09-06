// Simple math operations to be used with device_patterns.cuh

// TODO(ussuri): Why are floats passed by reference here? 

#include <ratio>

struct identity_op {
  __host__ __device__ static inline constexpr float op(float a) {
    return a;
  }
};

template <typename scale>
struct scale_op {
  __host__ __device__ static inline float op(float a) {
    return a * scale::num / scale::den;
  }
};

// NOTE: Take `INCR` by ref b/c pure float template params are non-standard.
template <typename incr>
struct incr_op {
  __host__ __device__ static inline float op(float a) {
    return a + (1.0L * incr::num / incr::den);
  }
};

template <typename inner_op = identity_op>
struct square_op {
  __host__ __device__ static inline float op(float a) {
    const auto aa = inner_op::op(a);
    return aa * aa;
  }

  __host__ __device__ static inline float op(float a, float b) {
    const auto aa = inner_op::op(a, b);
    return aa * aa;
  }
};

template <typename inner_op = identity_op>
struct sinh_op {
  __host__ __device__ static inline float op(float a) {
    return std::sinh(inner_op::op(a));
  }
};

template <typename inner_op = identity_op>
struct square_root_op {
  __host__ __device__ static inline float op(float a) {
    return std::sqrt(inner_op::op(a));
  }
};

template <typename inner_op_a = identity_op, typename inner_op_b = identity_op>
struct add_op {
  __host__ __device__ static inline float op(float a, float b) {
    return inner_op_a::op(a) + inner_op_b::op(b);
  }

  // Init value for reduction use of this op
  __host__ __device__ static inline float init() {
    return 0.0;
  }
};

struct mul_op {
  __host__ __device__ static inline float op(float a, float b) {
    return a * b;
  }
};

template <typename inner_op = identity_op>
struct div_op {
  __host__ __device__ static inline float op(float a, float b) {
    return inner_op::op(a) / inner_op::op(b);
  }
};

struct sub_op {
  __host__ __device__ static inline float op(float a, float b) {
    return a - b;
  }
};
