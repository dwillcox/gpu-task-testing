#include <iostream>
#include <cassert>
#include "device_vector.h"

class State {
public:
  double x;
  int status;
  __host__ __device__ State() : x(0.0), status(0) {}
};

__global__ void double_tens(int size, double* xsum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  *xsum = 0.0;
  if (tid == 0) {
    DeviceVector<State> svec;
    svec.resize(size);
    for (int i = 0; i < size; i++) {
      svec.push_back(State());
    }
    for (int i = 0; i < svec.size(); i++) {
      svec[i].x = 10.0;
    }
    for (auto it = svec.begin(); it != svec.end(); ++it) {
      (*it).x *= 2;
    }    
    for (auto it = svec.begin(); it != svec.end(); ++it) {
      *xsum += (*it).x;
    }
  }
}

int main(int argc, char* argv[]) {

  double* xsum;
  int size = 100;

  cudaError_t cuda_status = cudaMallocManaged(&xsum, sizeof(double));
  assert(cuda_status == cudaSuccess);

  double_tens<<<1,1>>>(size, xsum);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  std::cout << "sum of 2*10 100 times is: " << *xsum << std::endl;

  if (*xsum == 2000.0) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
