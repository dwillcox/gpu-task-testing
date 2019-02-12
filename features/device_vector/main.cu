#include <iostream>
#include <cassert>
#include "device_vector.h"

class State {
public:
  DeviceVector<double> x;
  int size;
  double xsum;
  __host__ __device__ State() {}
};

__global__ void double_tens(State* state) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    state->xsum = 0.0;
    state->x.resize(state->size);
    for (int i = 0; i < state->size; i++) {
      state->x.push_back(10.0);
    }
    for (auto it = state->x.begin(); it != state->x.end(); ++it) {
      (*it) *= 2;
    }    
    for (auto it = state->x.begin(); it != state->x.end(); ++it) {
      state->xsum += (*it);
    }
  }
}

int main(int argc, char* argv[]) {

  State* state;
  int size = 100;

  cudaError_t cuda_status = cudaMallocManaged(&state, sizeof(State));
  assert(cuda_status == cudaSuccess);

  state->size = size;

  double_tens<<<1,1>>>(state);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  std::cout << "sum of 2*10 100 times is: " << state->xsum << std::endl;

  if (state->xsum == 2000.0) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
