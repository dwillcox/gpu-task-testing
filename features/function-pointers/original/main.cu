#include <iostream>
#include <cassert>

__device__
void cube(double* xi) {
  *xi = (*xi) * (*xi) * (*xi);
}

__global__
void cube_kernel(double* x, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    cube(&x[tid]);
  }
}


int main(int argc, char* argv[]) {

  double* x = NULL;
  int size = 100;
  cudaError_t cuda_status = cudaSuccess;

  cuda_status = cudaMallocManaged(&x, sizeof(double) * size);
  assert(cuda_status == cudaSuccess);

  for (int i = 0; i < size; i++) {
    x[i] = 2.0;
  }
  
  int numThreads = std::min(32, size);
  int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

  cube_kernel<<<numBlocks, numThreads>>>(x, size);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  double xsum = 0.0;
  for (int i = 0; i < size; i++) {
    xsum += x[i];
  }

  std::cout << "sum of elementwise cubed x is: " << xsum << std::endl;

  if (xsum == 800.0) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
