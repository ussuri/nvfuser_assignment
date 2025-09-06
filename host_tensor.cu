#include "host_tensor.cuh"

// DO NOT modify any code in this file
template <int N_DIMS>
void host_tensor<N_DIMS>::fill_random() {
  srand(static_cast<int>(time(NULL)));
  for (size_t i = 0; i < this->get_n_elems(); i++) {
    this->get()[i] = static_cast<float>(rand() / RAND_MAX * 2.0 - 1.0);
  }
}

template <int N_DIMS>
void host_tensor<N_DIMS>::fill(float val) {
  for (size_t i = 0; i < this->get_n_elems(); i++) {
    this->get()[i] = val;
  }
}

template <int N_DIMS>
void host_tensor<N_DIMS>::copy(const device_tensor<N_DIMS>& other) {
  assert(this->get_n_elems() == other.get_n_elems());
  CHECK(cudaMemcpy(
      this->get(),
      other.get(),
      this->get_n_elems() * sizeof(float),
      cudaMemcpyDeviceToHost));
};

template <int N_DIMS>
void host_tensor<N_DIMS>::copy(const host_tensor<N_DIMS>& other) {
  assert(this->get_n_elems() == other.get_n_elems());
  for (int i = 0; i < this->get_n_elems(); i++)
    this->get()[i] = other.get()[i];
};

// Instantiate
template class host_tensor<1>;
template class host_tensor<2>;
template class host_tensor<3>;
