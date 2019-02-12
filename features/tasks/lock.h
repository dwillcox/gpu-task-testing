#ifndef LOCK_H
#define LOCK_H
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
#endif
