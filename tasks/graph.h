#ifndef GRAPH_H
#define GRAPH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "pool.h"
#include "task.h"
#include "unified.h"
#include "generic_function.h"

__global__ void device_execute_graph(void* vgraph);

class Graph : public UnifiedMemoryClass {
public:
  thrust::device_vector<Pool*> task_pools;
  thrust::device_vector<GenericFunction*> generic_function_table;
  cudaStream_t* pool_streams;

  // doesn't need a lock because only the device will modify
  // and the host will check it in a loop.
  bool active_device_kernels = false;
  
  Graph() {
    initialize_function_tables();
    initialize_task_pools();
  }

  ~Graph() {
    thrust::host_vector<Pool*> host_task_pools = task_pools;
    for (auto p : host_task_pools) {
      delete p;
    }
    for (auto gf : generic_function_table) {
      delete gf;
    }
  }
  
  void initialize_function_tables();

  void initialize_task_pools() {
    thrust::host_vector<Pool*> host_pools;
    Pool* p;
    for (auto gf : generic_function_table) {
      p = new Pool();
      host_pools.push_back(p);
    }
    task_pools = host_pools;
  }

  void queue(void* state) {
    GenericFunction* gf = generic_function_table[0];
    Task* t = new Task(gf, state);
    Pool* p = task_pools[0];
    p->checkin(t);
  }

  // for now, advance is responsible for deleting all Tasks
  // created by queue to avoid memory leaks. fix for generality.
  __host__ __device__ void advance(thrust::device_vector<Task*>* task_collection);

  void execute_graph() {
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // execute host and device kernels for the graph
    active_device_kernels = true;
    device_execute_graph<<<1,1>>>(static_cast<void*>(this));

    // !! Change this to only check task pools supporting host execution
    thrust::device_vector<Task*> host_task_container;
    while (active_device_kernels) {
      for (Pool* p : task_pools) {
	p->checkout(&host_task_container);
	for (Task* t_p : host_task_container) {
	  t_p->execute();
	}
	p->report_kernel_finished();
      }
    }

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
  }
};
#endif
