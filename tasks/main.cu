#include <iostream>
#include <cassert>
#include "unified.h"
#include "task.h"
#include "lock.h"
#include "generic_function.h"
#include "graph.h"
#include "pool.h"
#include "state.h"
#include "device_vector.h"

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

__global__ void task_kernel(DeviceVector<Task*>* task_container) {
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
    DeviceVector<Task*>* task_container = new DeviceVector<Task*>();
    while (!task_pool->signal_shutdown) {
      task_pool->checkout(task_container);
      int size = task_container->size();
      if (size > 0) {
	// set active
	graph_ptr->pool_active_status[task_pool->pool_graph_index] = 1;

	int numThreads = min(32, size);
	int numBlocks = static_cast<int>(ceil(((double) size)/((double) numThreads)));

	task_kernel<<<numBlocks, numThreads, 0, *(task_pool->stream_ptr)>>>(task_container);

	graph_ptr->advance(task_container);
      } else {
	// set inactive
	graph_ptr->pool_active_status[task_pool->pool_graph_index] = 0;
      }
      task_pool->report_kernel_finished();
    }
    delete[] task_container;
  }
}

__global__
void device_execute_graph(void* vgraph) {
  Graph* graph = static_cast<Graph*>(vgraph);
  
  // by construction there will be just as many threads as task pools
  // launch a kernel in a unique stream for every task pool  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  Pool* p;
  p = graph->task_pools[tid];
  p->set_stream(&(graph->pool_streams[tid]));
  pool_kernel<<<1,1,0,graph->pool_streams[tid]>>>(p, graph);
  __syncthreads();

  // wait for all the tasks to be finished
  if (tid == 0) {
    bool graph_finished = true;
    do {
      graph_finished = true;
      for (int ipool = 0; ipool < graph->num_task_pools; ipool++) {
	if (graph->pool_active_status[ipool] == 1) graph_finished=false;
      }
    } while (!graph_finished);

    // send kernels shutdown signal and stream sync
    for (Pool* p : graph->task_pools) {
      p->post_shutdown_signal();
    }

    graph->active_device_kernels = false;    
  }
  __syncthreads();
}

void Graph::initialize_function_tables() {
  execute_task_t* fun_ptr = nullptr;
  GenericFunction* gf;

  cudaError_t cuda_status = cudaSuccess;
  
  get_cube_pointer<<<1,1>>>(fun_ptr);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
  gf = new GenericFunction(nullptr, fun_ptr);
  generic_function_table_host.push_back(gf);
  
  get_square_pointer<<<1,1>>>(fun_ptr);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
  gf = new GenericFunction(nullptr, fun_ptr);
  generic_function_table_host.push_back(gf);

  *fun_ptr = square_host;
  gf = new GenericFunction(fun_ptr, nullptr);
  generic_function_table_host.push_back(gf);

  generic_function_table_device.copy_host_to_device(&generic_function_table_host);

  num_task_pools = 3;
}

__host__ __device__ void Graph::advance(DeviceVector<Task*>* task_collection) {
  for (Task* task : *task_collection) {
    State* state = static_cast<State*>(task->state);
    if (state->status == 0) {
      #ifdef __CUDA_ARCH__
      task->gfun = generic_function_table_device[1];
      Pool* p= task_pools[1];      
      #else
      task->gfun = generic_function_table_host[1];
      Pool* p= task_pools_host[1];
      #endif
      state->status = 1;
      p->checkin(task);
      pool_active_status[1] = 1;
    } else if (state->status == 1) {
      #ifdef __CUDA_ARCH__
      task->gfun = generic_function_table_device[2];
      Pool* p= task_pools[2];
      #else
      task->gfun = generic_function_table_host[2];
      Pool* p= task_pools_host[2];
      #endif      
      state->status = 2;
      p->checkin(task);
      pool_active_status[2] = 1;
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
