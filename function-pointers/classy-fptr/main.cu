#include <iostream>
#include <cassert>

typedef void (* execute_task_t)(void *);

class CubeTask {
  double* xi;
  execute_task_t* cube_on_device;
public:
  __host__ __device__ CubeTask();
  __host__ __device__ CubeTask(execute_task_t* cube_fp, double* x) : cube_on_device(cube_fp), xi(x) {}
  __device__ void execute() {
    (*cube_on_device)(xi);
  }
};

__device__
void cube(void* xv) {
  double* xi = static_cast<double*>(xv);
  *xi = (*xi) * (*xi) * (*xi);
}

__global__
void task_kernel(CubeTask* tasks, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    tasks[tid].execute();
  }
}

__global__
void get_cube_pointer(execute_task_t* device_pointer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  
  if (tid == 0) *device_pointer = cube;
}

int main(int argc, char* argv[]) {

  double* x = NULL;
  int size = 100;

  execute_task_t* device_cube_fun_p = NULL;
  CubeTask* tasks = NULL;
  
  cudaError_t cuda_status = cudaSuccess;

  cuda_status = cudaMallocManaged(&x, sizeof(double) * size);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&device_cube_fun_p, sizeof(execute_task_t));
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&tasks, sizeof(CubeTask)*size);
  assert(cuda_status == cudaSuccess);

  get_cube_pointer<<<1,1>>>(device_cube_fun_p);

  for (int i = 0; i < size; i++) {
    x[i] = 2.0;
  }

  for (int i = 0; i < size; i++) {
    tasks[i] = CubeTask(device_cube_fun_p, &x[i]);
  }

  int numThreads = std::min(32, size);
  int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

  task_kernel<<<numBlocks, numThreads>>>(tasks, size);

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
