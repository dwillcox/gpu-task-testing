#include <iostream>
#include <cassert>

typedef void (* execute_task_t)(void *);

class UnifiedMemoryClass
{
public:
  void* operator new(size_t size) {
    void* vp;
    cudaMallocManaged(&vp, size);
    cudaDeviceSynchronize();
    return vp;
  }

  void operator delete(void* vp) {
    cudaDeviceSynchronize();
    cudaFree(vp);
  }
};

class UnaryTask {
  double* xi;
  execute_task_t* unary_on_device;
public:
  __host__ __device__ UnaryTask();
  __host__ __device__ UnaryTask(execute_task_t* unary_device_fp = NULL, double* x = NULL) : unary_on_device(unary_device_fp), xi(x) {}
  __device__ void execute() {
    (*unary_on_device)(xi);
  }
};

__global__
void device_mutex_lock(int* mutex) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) while(atomicCAS(mutex, 0, 1) != 0);
}

__global__
void device_mutex_unlock(int* mutex) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) atomicExch(mutex, 0);
}

class HostDeviceLock : public UnifiedMemoryClass {
  int* mutex;
  cudaStream_t lockStream;
public:
  HostDeviceLock() {
    cudaError_t cuda_status = cudaSuccess;
    cuda_status = cudaMalloc((void**) &mutex, sizeof(int));
    assert(cuda_status == cudaSuccess);

    int unlocked = 0;
    cuda_status = cudaMemcpy(mutex, &unlocked, sizeof(int), cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaStreamCreate(&lockStream);
    assert(cuda_status == cudaSuccess);
  }

  __host__ __device__
  void lock() {
    #ifdef __CUDA_ARCH__
    while(atomicCAS(mutex, 0, 1) != 0);
    #else
    device_mutex_lock<<<1,1,0,lockStream>>>(mutex);
    cudaError_t cuda_status = cudaStreamSynchronize(lockStream);
    assert(cuda_status == cudaSuccess);
    #endif
  }

  __host__ __device__
  void unlock() {
    #ifdef __CUDA_ARCH__
    atomicExch(mutex, 0);
    #else
    device_mutex_unlock<<<1,1,0,lockStream>>>(mutex);
    cudaError_t cuda_status = cudaStreamSynchronize(lockStream);
    assert(cuda_status == cudaSuccess);    
    #endif
  }
};

__device__
void cube(void* xv) {
  double* xi = static_cast<double*>(xv);
  *xi = (*xi) * (*xi) * (*xi);
}

__device__
void square(void* xv) {
  double* xi = static_cast<double*>(xv);
  *xi = (*xi) * (*xi);
}

__global__
void task_kernel(UnaryTask* tasks, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    tasks[tid].execute();
  }
}

__global__
void task_launcher(UnaryTask* tasks, int size, HostDeviceLock* task_lock) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) {
    task_lock->lock();

    int numThreads = min(32, size);
    int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

    task_kernel<<<numBlocks, numThreads>>>(tasks, size);
    cudaDeviceSynchronize();

    task_lock->unlock();
  }
}

__global__
void get_cube_pointer(execute_task_t* device_pointer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  
  if (tid == 0) *device_pointer = cube;
}

__global__
void get_square_pointer(execute_task_t* device_pointer) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;  
  if (tid == 0) *device_pointer = square;
}

void square_host(double* x, int size, HostDeviceLock* task_lock) {
  task_lock->lock();
  for (int i = 0; i < size; i++) {
    x[i] = x[i] * x[i];
  }
  task_lock->unlock();
}

int main(int argc, char* argv[]) {

  double* x = NULL;
  int size = 100;

  execute_task_t* device_cube_fun_p = NULL;
  execute_task_t* device_square_fun_p = NULL;
  
  UnaryTask* cube_tasks = NULL;
  UnaryTask* square_tasks = NULL;

  cudaStream_t streamA;
  cudaStream_t streamB;

  cudaError_t cuda_status = cudaSuccess;

  HostDeviceLock* task_lock = new HostDeviceLock();

  cuda_status = cudaStreamCreate(&streamA);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaStreamCreate(&streamB);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&x, sizeof(double) * size);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&device_cube_fun_p, sizeof(execute_task_t));
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&cube_tasks, sizeof(UnaryTask)*size);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&device_square_fun_p, sizeof(execute_task_t));
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&square_tasks, sizeof(UnaryTask)*size);
  assert(cuda_status == cudaSuccess);

  get_cube_pointer<<<1,1>>>(device_cube_fun_p);
  get_square_pointer<<<1,1>>>(device_square_fun_p);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  for (int i = 0; i < size; i++) {
    x[i] = 2.0;
  }

  for (int i = 0; i < size; i++) {
    square_tasks[i] = UnaryTask(device_square_fun_p, &x[i]);
  }

  for (int i = 0; i < size; i++) {
    cube_tasks[i] = UnaryTask(device_cube_fun_p, &x[i]);
  }

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
  
  int numThreads = min(32, size);
  int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

  task_launcher<<<1,1,0,streamA>>>(square_tasks, size, task_lock);
  task_launcher<<<1,1,0,streamB>>>(cube_tasks, size, task_lock);
  square_host(x, size, task_lock);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);

  double xsum = 0.0;
  for (int i = 0; i < size; i++) {
    xsum += x[i];
  }

  std::cout << "sum of elementwise squared cubed squared x is: " << xsum << std::endl;

  if (xsum == 409600.0) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
