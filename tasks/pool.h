#ifndef POOL_H
#define POOL_H
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include "task.h"
#include "lock.h"
#include "unified.h"
#include "graph.h"

// As a first pass, let's use Pool-wide locking
class Pool : public UnifiedMemoryClass {
  thrust::device_vector<Task*> device_tasks;
  HostDeviceLock* lock;
  int num_running_kernels;
public:
  bool signal_shutdown;
  cudaStream_t* stream_ptr;  
  
  Pool() {
    lock = new HostDeviceLock();
    signal_shutdown = false;
    num_running_kernels = 0;
  }

  ~Pool() {
    delete lock;
    // delete all tasks we know about (there should not be any) or error maybe
  }

  __host__ __device__ void checkin(Task* task) {
    lock->lock();
    device_tasks.push_back(task);
    lock->unlock();
  }

  __host__ __device__ void checkout(thrust::device_vector<Task*>* task_container) {
    lock->lock();
    task_container->resize(0);

    thrust::device_vector<Task*> scratch;
    
    #ifdef __CUDA_ARCH__
    for (Task* t_p : device_tasks) {
      if (t_p->gfun->supports_device()) {
	task_container->push_back(t_p);
      } else {
	scratch.push_back(t_p);
      }
    }
    device_tasks = scratch;
    #else
    for (Task* t_p : device_tasks) {
      if (t_p->gfun->supports_host()) {
	task_container->push_back(t_p);
      } else {
	scratch.push_back(t_p);
      }
    }
    device_tasks = scratch;
    #endif

    num_running_kernels++;
    lock->unlock();
  }

  __host__ __device__ void report_kernel_finished() {
    lock->lock();
    num_running_kernels--;
    lock->unlock();
  }

  __host__ __device__ bool is_inactive() {
    return (device_tasks.size() == 0 && num_running_kernels == 0);
  }

  __host__ __device__ void set_stream(cudaStream_t* cuda_stream_ptr) {
    stream_ptr = cuda_stream_ptr;
  }

  __host__ __device__ void post_shutdown_signal() {
    lock->lock();
    signal_shutdown = true;
    lock->unlock();
  }
};
#endif
