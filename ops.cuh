// Simple math operations to be used with device_patterns.cuh

// TODO(ussuri): Why are floats passed by reference here? 

#include <ratio>

struct identity_op {
  __host__ __device__ static inline float op(const float& a) {
    return a;
  }
};

template <int NUM = 1, int DENOM = 1>
struct scale_op {
  __host__ __device__ static inline float op(const float& a) {
    return a * NUM / DENOM;
  }
};

// NOTE: Must take `INCR` by ref b/c pure float template params are non-standard.
template <const float& INCR>
struct incr_op {
  __host__ __device__ inline float op(const float& a) {
    return a + INCR;
  }
};

template <typename inner_op = identity_op>
struct square_op {
  __host__ __device__ static inline float op(const float& a) {
    const auto aa = inner_op::op(a);
    return aa * aa;
  }

  __host__ __device__ static inline float op(const float& a, const float& b) {
    const auto aa = inner_op::op(a, b);
    return aa * aa;
  }
};

template <typename inner_op = identity_op>
struct sinh_op {
  __host__ __device__ static inline float op(const float& a) {
    return std::sinh(inner_op::op(a));
  }
};

template <typename inner_op = identity_op>
struct square_root_op {
  __host__ __device__ static inline float op(const float& a) {
    return std::sqrt(inner_op::op(a));
  }
};

template <typename inner_op = identity_op>
struct add_op {
  __host__ __device__ static inline float op(const float& a, const float& b) {
    return inner_op::op(a) + inner_op::op(b);
  }

  // Init value for reduction use of this op
  __host__ __device__ static inline float init() {
    return 0.0;
  }
};

struct mul_op {
  __host__ __device__ static inline float op(const float& a, const float& b) {
    return a * b;
  }
};

template <typename inner_op = identity_op>
struct div_op {
  __host__ __device__ static inline float op(const float& a, const float& b) {
    return inner_op::op(a) / inner_op::op(b);
  }
};

struct sub_op {
  __host__ __device__ static inline float op(const float& a, const float& b) {
    return a - b;
  }
};
