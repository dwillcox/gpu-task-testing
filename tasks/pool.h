#ifndef POOL_H
#define POOL_H
#include <vector>
#include "device_vector.h"
#include "task.h"
#include "lock.h"
#include "unified.h"
#include "graph.h"

// As a first pass, let's use Pool-wide locking
class Pool : public UnifiedMemoryClass {
  DeviceVector<Task*> device_tasks;
  HostDeviceLock* lock;
  int num_running_kernels;
public:
  int pool_graph_index;  
  bool signal_shutdown;
  cudaStream_t* stream_ptr;  
  
  Pool(int index) {
    lock = new HostDeviceLock();
    signal_shutdown = false;
    num_running_kernels = 0;
    pool_graph_index = index;
  }

  ~Pool() {
    delete lock;
  }

  __host__ __device__ void checkin(Task* task) {
    lock->lock();
    #ifdef __CUDA_ARCH__
    device_tasks.push_back(task);
    #else
    DeviceVector_push_back<Task*><<<1,1>>>(&device_tasks, task);
    #endif
    lock->unlock();
  }

  __host__ __device__ void checkout(DeviceVector<Task*>* task_container) {
    lock->lock();

    #ifdef __CUDA_ARCH__
    DeviceVector<Task*> scratch;    
    task_container->resize(0);    
    for (Task* t_p : device_tasks) {
      if (t_p->gfun->supports_device()) {
	task_container->push_back(t_p);
      } else {
	scratch.push_back(t_p);
      }
    }
    device_tasks = scratch;
    #else
    std::vector<Task*> scratch;
    std::vector<Task*> device_tasks_h;
    std::vector<Task*> task_container_h;
    device_tasks.copy_device_to_host(&device_tasks_h);
    for (Task* t_p : device_tasks_h) {
      if (t_p->gfun->supports_host()) {
	task_container_h.push_back(t_p);
      } else {
	scratch.push_back(t_p);
      }
    }
    task_container->copy_host_to_device(&task_container_h);
    device_tasks.copy_host_to_device(&scratch);
    #endif
    num_running_kernels++;
    lock->unlock();
  }

  __host__ __device__ void report_kernel_finished() {
    lock->lock();
    num_running_kernels--;
    lock->unlock();
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
