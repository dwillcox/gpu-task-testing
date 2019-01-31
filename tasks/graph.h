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
  DeviceVector<Pool*> task_pools;
  std::vector<Pool*> task_pools_host;
  
  DeviceVector<GenericFunction*> generic_function_table_device;
  std::vector<GenericFunction*> generic_function_table_host;
  
  std::vector<Task*> task_list;
  cudaStream_t* pool_streams;

  //!! this could have race conditions between pool reporting and graph checking
  int* pool_active_status;

  int num_task_pools;

  // doesn't need a lock because only the device will modify
  // and the host will check it in a loop.
  bool active_device_kernels = false;
  bool streams_allocated = false;
  
  Graph() : num_task_pools(0) {
    initialize_function_tables();
    initialize_task_pools();
  }

  ~Graph() {
    for (auto p : task_pools_host) {
      delete p;
    }
    for (auto gf : generic_function_table_host) {
      delete gf;
    }
    for (Task* t : task_list) {
      delete t;
    }
    cudaError_t cuda_status = cudaFree(pool_active_status);
    assert(cuda_status == cudaSuccess);
  }
  
  void initialize_function_tables();

  void initialize_task_pools() {
    Pool* p;
    cudaError_t cuda_status = cudaMallocManaged(&pool_active_status, sizeof(int)*num_task_pools);
    assert(cuda_status == cudaSuccess);
    for (int i = 0; i < num_task_pools; i++) {
      p = new Pool(i);
      task_pools_host.push_back(p);
      pool_active_status[i] = 0;
    }
    task_pools.copy_host_to_device(&task_pools_host);
  }

  void queue(void* state) {
    GenericFunction* gf = generic_function_table_host[0];
    Task* t = new Task(gf, state);
    task_list.push_back(t);
    Pool* p = task_pools_host[0];
    p->checkin(t);
  }

  // for now, advance is responsible for deleting all Tasks
  // created by queue to avoid memory leaks. fix for generality.
  __host__ __device__ void advance(DeviceVector<Task*>* task_collection);

  void execute_graph() {
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // create 1 CUDA stream per task pool
    cuda_status = cudaMallocManaged(&pool_streams, num_task_pools*sizeof(cudaStream_t));
    assert(cuda_status == cudaSuccess);
    for (int i = 0; i < num_task_pools; i++) {
      cuda_status = cudaStreamCreate(&(pool_streams[i]));
      assert(cuda_status == cudaSuccess);      
    }
    streams_allocated = true;

    // execute host and device kernels for the graph
    // use 1 block and 1 thread per task pool so we can use __syncthreads
    active_device_kernels = true;

    // this can't reasonably be bigger than 1024 or reconsider this strategy
    int numThreads = num_task_pools;
    int numBlocks = 1;
    device_execute_graph<<<numBlocks,numThreads>>>(static_cast<void*>(this));

    // !! commented because DeviceVector doesn't support this ...
    // // !! Change this to only check task pools supporting host execution
    // thrust::device_vector<Task*> host_task_container;
    // while (active_device_kernels) {
    //   for (Pool* p : task_pools) {
    // 	p->checkout(&host_task_container);
    // 	for (Task* t_p : host_task_container) {
    // 	  t_p->execute();
    // 	}
    // 	p->report_kernel_finished();
    //   }
    // }

    // sync streams
    for (int i = 0; i < num_task_pools; i++) {
      cuda_status = cudaStreamSynchronize(pool_streams[i]);
      assert(cuda_status == cudaSuccess);
    }

    // is this really needed???
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    for (int i = 0; i < num_task_pools; i++) {
      cuda_status = cudaStreamDestroy(pool_streams[i]);
      assert(cuda_status == cudaSuccess);
    }    
  }
};
#endif
