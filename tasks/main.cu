#include <iostream>
#include <cassert>
#include "unified.h"
#include "lock.h"
#include "graph.h"
#include "pool.h"
#include "state.h"
#include "generic_vector.h"

void square_host(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
}

__global__ void pool_kernel(Pool* pool) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = pool->checked_out_tasks.size();
  if (tid < size) {
    GenericVector<State*>* vptr = &pool->checked_out_tasks;
    State** s = vptr->get_device_pointer_to(tid);
    (*s)->advance();
  }
}

void Graph::advance(GenericVector<State*>& advance_states) {
  std::cout << "in advance ..." << std::endl;
  for (State* state : advance_states) {
    if (state->status == 0) {
      Pool** p = device_task_pools.get_pointer_to(0);
      (*p)->checkin(state);
    }
    else if (state->status == 1) {
      Pool** p = device_task_pools.get_pointer_to(1);
      (*p)->checkin(state);
    }
    else if (state->status == 2) {
      Pool** p = host_task_pools.get_pointer_to(0);
      (*p)->checkin(state);
    }
  }
  std::cout << "leaving advance ..." << std::endl;  
}


int main(int argc, char* argv[]) {

  State* state = NULL;
  int size = 100;

  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaDeviceSynchronize();
  std::cout << "device status before Graph: " << cudaGetErrorString(cuda_status) << std::endl;

  cuda_status = cudaMallocManaged(&state, sizeof(State)*size);
  assert(cuda_status == cudaSuccess);

  // create a Graph with 1 host task pool and 2 device task pools
  Graph* task_graph = new Graph(1, 2);

  cuda_status = cudaDeviceSynchronize();
  std::cout << "device status after Graph: " << cudaGetErrorString(cuda_status) << std::endl;

  for (int i = 0; i < size; i++) {
    state[i].x = 2.0;
    if (i > 20 && i < 70)
      state[i].status = 1;
    else if (i <= 20)
      state[i].status = 0;
    else
      state[i].status = 2;
    task_graph->queue(&(state[i]));
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
