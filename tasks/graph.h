#ifndef GRAPH_H
#define GRAPH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "pool.h"
#include "state.h"
#include "unified.h"
#include "generic_vector.h"

__global__ void pool_kernel(Pool* pool);

class Graph : public UnifiedMemoryClass {
public:
  GenericVector<Pool*> device_task_pools;
  GenericVector<Pool*> host_task_pools;  
  GenericVector<State*> task_registry;
  cudaStream_t* pool_streams;

  bool graph_finished;
  
  Graph(size_t nhostp, size_t ndevp) : graph_finished(false) {
    initialize_task_pools(nhostp, ndevp);
    std::cout << "initialized task pools" << std::endl;
  }

  ~Graph() {
    cudaError_t cuda_status = cudaSuccess;
    for (Pool* p : device_task_pools) {
      cuda_status = cudaStreamDestroy(p->get_stream());
      assert(cuda_status == cudaSuccess);      
    }
    cuda_status = cudaFree(pool_streams);
    assert(cuda_status == cudaSuccess);
    
    for (Pool* p : device_task_pools) {
      delete p;
    }
    
    for (Pool* p : host_task_pools) {
      delete p;
    }
    
    for (State* t : task_registry) {
      delete t;
    }
  }

  void initialize_task_pools(size_t num_host_pools, size_t num_device_pools) {
    Pool* p;
    
    for (int i = 0; i < num_host_pools; i++) {
      p = new Pool(i);
      p->set_host_pool();
      host_task_pools.push_back(p);
    }
    
    for (int i = 0; i < num_device_pools; i++) {
      p = new Pool(i);
      p->set_device_pool();
      device_task_pools.push_back(p);
    }

    // create 1 CUDA stream per task pool and attach it to the pool
    cudaError_t cuda_status = cudaMallocManaged(&pool_streams, device_task_pools.size()*sizeof(cudaStream_t));
    assert(cuda_status == cudaSuccess);
    for (int i = 0; i < device_task_pools.size(); i++) {
      cuda_status = cudaStreamCreate(&(pool_streams[i]));
      assert(cuda_status == cudaSuccess);
      device_task_pools[i]->set_stream(&(pool_streams[i]));
    }

    device_task_pools.sync_to_device();    
  }

  void queue(State* state) {
    task_registry.push_back(state);
  }

  bool completed() {
    bool tasks_unfinished = false;
    for (State* state : task_registry) {
      if (state->status != 3) {
	tasks_unfinished = true;
	break;
      }
    }
    graph_finished = !tasks_unfinished;
    return graph_finished;
  }
  
  void advance(GenericVector<State*>& advance_states);

  void execute_graph() {
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // Initialize task pools with queued tasks in the registry    
    std::cout << "initializing task pools..." << std::endl;
    advance(task_registry);

    std::cout << "starting graph execution..." << std::endl;    
    
    while (!completed()) {

      // check if previous device pool kernels finished and advance states
      int i = 0;
      for (Pool* pool : device_task_pools) {
	if (pool->finished()) {
	  advance(pool->checked_out_tasks);
	  pool->reset_checked_out_tasks();
	}
	i++;
      }

      // launch device task kernels for pools that are ready
      i = 0;
      for (Pool* pool : device_task_pools) {      
	if (pool->ready()) {
	  int ntasks = pool->size_queued();
	  std::cout << "got " << ntasks << " ntasks for device pool " << i << std::endl;	  
	  int numThreads = min(32, ntasks);
	  int numBlocks = static_cast<int>(ceil(((double) ntasks)/((double) numThreads)));

	  // checkout and copy task pointers to the device
	  pool->checkout();
	  pool->checked_out_tasks.sync_to_device();
	  pool->set_active();
	  
	  // launch kernel
	  pool_kernel<<<numBlocks, numThreads, 0, pool->get_stream()>>>(pool);
	  cudaEventRecord(pool->kernel_finished, pool->get_stream());
	}
	i++;
      }

      // check if previous host pool kernels finished and advance states      
      i = 0;
      for (Pool* pool : host_task_pools) {
	if (pool->finished()) {
	  advance(pool->checked_out_tasks);
	  pool->reset_checked_out_tasks();
	}
	i++;
      }

      // execute host tasks
      i = 0;
      for (Pool* pool : host_task_pools) {
	if (pool->ready()) {
	  int ntasks = pool->size_queued();
	  std::cout << "got " << ntasks << " ntasks for host pool " << i << std::endl;

	  // checkout
	  pool->checkout();
	  pool->set_active();

	  // run batched tasks
	  State::batched_advance(pool->checked_out_tasks);

	  pool->set_inactive();

	  i++;
	}
      }
    }
    
    // sync device
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
  }
};
#endif
