#include <iostream>
#include <cassert>
#include "unified.h"
#include "task.h"
#include "lock.h"
#include "generic_function.h"
#include "graph.h"
#include "pool.h"
#include "state.h"

__device__
void cube(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x) * (state->x);
}

__device__
void square(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
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

void square_host(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
}

__global__ void task_kernel(thrust::device_vector<Task*>* task_container) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = task_container->size();
  if (tid < size) {
    Task* task_p = (*task_container)[tid];
    task_p->execute();
  }
}

__global__ void pool_kernel(Pool* task_pool, Graph* graph_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    thrust::device_vector<Task*> task_container;
    while (!task_pool->signal_shutdown) {
      task_pool->checkout(&task_container);
      int size = task_container.size();
      if (size > 0) {
	int numThreads = min(32, size);
	int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

	task_kernel<<<numBlocks, numThreads, 0, *(task_pool->stream_ptr)>>>(&task_container);
	cudaStreamSynchronize(*(task_pool->stream_ptr));

	graph_ptr->advance(&task_container);
      }
      task_pool->report_kernel_finished();
    }
  }
}

__global__
void device_execute_graph(void* vgraph) {
  Graph* graph = static_cast<Graph*>(vgraph);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // launch a kernel in a unique stream for every task pool
    int npools = graph->task_pools.size();    
    graph->pool_streams = (cudaStream_t*) malloc(npools*sizeof(cudaStream_t));
    Pool* p;
    for (int i = 0; i < npools; i++) {
      cudaStreamCreate(&(graph->pool_streams[i]));
      p = graph->task_pools[i];
      p->set_stream(&(graph->pool_streams[i]));
      pool_kernel<<<1,1,0,graph->pool_streams[i]>>>(p, graph);
    }

    // wait for all the tasks to be finished
    bool graph_finished = true;
    do {
      graph_finished = true;
      for (Pool* p : graph->task_pools) {
	if (!p->is_inactive()) {
	  graph_finished = false;
	}
      }
    } while (!graph_finished);

    // send kernels shutdown signal and stream sync
    for (Pool* p : graph->task_pools) {
      p->post_shutdown_signal();
      cudaStreamSynchronize(*(p->stream_ptr));
    }

    // destroy streams
    for (int i = 0; i < npools; i++) {
      cudaStreamDestroy(graph->pool_streams[i]);
    }

    graph->active_device_kernels = false;
  }
}

void Graph::initialize_function_tables() {
  execute_task_t* fun_ptr = nullptr;
  thrust::host_vector<GenericFunction*> generic_functions;
  GenericFunction* gf;

  cudaError_t cuda_status = cudaSuccess;
  
  get_cube_pointer<<<1,1>>>(fun_ptr);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
  gf = new GenericFunction(nullptr, fun_ptr);
  generic_functions.push_back(gf);
  
  get_square_pointer<<<1,1>>>(fun_ptr);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
  gf = new GenericFunction(nullptr, fun_ptr);
  generic_functions.push_back(gf);

  *fun_ptr = square_host;
  gf = new GenericFunction(fun_ptr, nullptr);
  generic_functions.push_back(gf);

  generic_function_table = generic_functions;
}

__host__ __device__ void Graph::advance(thrust::device_vector<Task*>* task_collection) {
  for (Task* task : *task_collection) {
    State* state = static_cast<State*>(task->state);
    if (state->status == 0) {
      task->gfun = generic_function_table[1];
      state->status = 1;
      Pool* p= task_pools[1];
      p->checkin(task);
    } else if (state->status == 1) {
      task->gfun = generic_function_table[2];
      state->status = 2;
      Pool* p= task_pools[2];      
      p->checkin(task);
    }
  }
}

int main(int argc, char* argv[]) {

  State* state = NULL;
  int size = 100;

  Graph* task_graph = new Graph();

  cudaStream_t streamA;
  cudaStream_t streamB;

  cudaError_t cuda_status = cudaSuccess;

  cuda_status = cudaStreamCreate(&streamA);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaStreamCreate(&streamB);
  assert(cuda_status == cudaSuccess);

  cuda_status = cudaMallocManaged(&state, sizeof(State) * size);
  assert(cuda_status == cudaSuccess);

  for (int i = 0; i < size; i++) {
    state[i].x = 2.0;
    state[i].status = 0;
    task_graph->queue(&state[i]);
  }

  task_graph->execute_graph();
  
  double xsum = 0.0;
  for (int i = 0; i < size; i++) {
    xsum += state[i].x;
  }

  std::cout << "sum of elementwise squared cubed squared x is: " << xsum << std::endl;

  if (xsum == 409600.0) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
