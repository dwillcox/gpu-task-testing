#include <iostream>
#include <cassert>
#include "generic_vector.h"
#include "unified.h"

class State : public UnifiedMemoryClass {
public:
  GenericVector<double> x;
  int size;
  int device_fill_size;
  double xsum;
  State() {
    //    x = new GenericVector<double>();
  }
};

__global__ void double_tens(State* state) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {

    for (int i = 0; i < state->x.size(); i++) {
      double* xi;
      xi = state->x.get_pointer_to(i);
      (*xi) *= 2;
    }    

    for (int i = 0; i < state->x.size(); i++) {
      double* xi;
      xi = state->x.get_pointer_to(i);      
      state->xsum += (*xi);
    }
  }
}

int main(int argc, char* argv[]) {
  State* state = new State();
  int size = 100;
  cudaError_t cuda_status = cudaSuccess;

  state->size = size;

  std::cout << "pushing data into vector on host ..." << std::endl;  
  for (int i = 0; i < size; i++) {
    state->x.push_back(10.0);
  }

  std::cout << "syncing vector to device ..." << std::endl;
  state->x.sync_to_device();

  std::cout << "launching work kernel ..." << std::endl;
  double_tens<<<1,1>>>(state);

  std::cout << "cuda status is " << cudaGetErrorString(cuda_status) << std::endl;  

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  std::cout << "sum of 2*10 " << size << " times is: " << state->xsum << std::endl;

  if (state->xsum == 2*10.0*size) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
