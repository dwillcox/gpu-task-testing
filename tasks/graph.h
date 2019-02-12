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
      host_task_pools.push_back(p);
    }
    
    for (int i = 0; i < num_device_pools; i++) {
      p = new Pool(i);
      device_task_pools.push_back(p);
    }

    // create 1 CUDA stream per task pool and attach it to the pool
    cudaError_t cuda_status = cudaMallocManaged(&pool_streams, device_task_pools.size()*sizeof(cudaStream_t));
    assert(cuda_status == cudaSuccess);
    for (int i = 0; i < device_task_pools.size(); i++) {
      cuda_status = cudaStreamCreate(&(pool_streams[i]));
      assert(cuda_status == cudaSuccess);
      device_task_pools[i]->set_cuda_stream(&(pool_streams[i]));
    }

    device_task_pools.sync_to_device();    
  }

  void queue(State* state) {
    task_registry.push_back(state);
  }

  void advance();

  void execute_graph() {
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // Initialize task pools with queued tasks in the registry
    advance();

    // this is very rough, in the future use events to do all this more asynchronously
    while (!graph_finished) {
      std::cout << "starting graph execution" << std::endl;

      // launch device task kernels
      int i = 0;
      for (Pool* pool : device_task_pools) {
	  int ntasks = pool->tasks.size();
	  std::cout << "got " << ntasks << " ntasks for device pool " << i << std::endl;

	  if (ntasks > 0) {
	    int numThreads = min(32, ntasks);
	    int numBlocks = static_cast<int>(ceil(((double) ntasks)/((double) numThreads)));

	    // copy task pointers to the device
	    pool->tasks.sync_to_device();
	    cuda_status = cudaDeviceSynchronize();
	    assert(cuda_status == cudaSuccess);	  
	  
	    pool_kernel<<<numBlocks, numThreads, 0, pool->get_stream()>>>(pool);	    
	  }
	  i++;
      }

      // execute host tasks
      i = 0;
      for (Pool* pool : host_task_pools) {
	int ntasks = pool->tasks.size();
	std::cout << "got " << ntasks << " ntasks for host pool " << i << std::endl;
	if (ntasks > 0) {
	  State::batched_advance(pool->tasks);
	}
	i++;
      }      

      // sync streams
      i = 0;
      for (Pool* pool : device_task_pools) {
      	cuda_status = cudaStreamSynchronize(pool->get_stream());
      	std::cout << "synchronizing stream " << i << " with result: " << cudaGetErrorString(cuda_status) << std::endl;
      	assert(cuda_status == cudaSuccess);
	std::cout << "clearing task pool ..." << std::endl;
	pool->reset();	
	i++;
      }

      std::cout << "calling advance ..." << std::endl;
      advance();
    }

    // sync device
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // destroy streams
    for (Pool* pool : device_task_pools) {
      cuda_status = cudaStreamDestroy(pool->get_stream());
      assert(cuda_status == cudaSuccess);      
    }
  }
};
#endif
