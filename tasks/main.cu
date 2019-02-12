#include <iostream>
#include <cassert>
#include <functional>
#include "unified.h"
#include "lock.h"
#include "graph.h"
#include "pool.h"
#include "state.h"
#include "unified_vector.h"


void square_host(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
}

__global__ void pool_kernel(Pool* pool) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = pool->checked_out_tasks.size();
  if (tid < size) {
      pool->checked_out_tasks[tid]->advance();
  }
}

void Graph::advance(UnifiedVector<State*>& advance_states) {
  std::cout << "in advance ..." << std::endl;

  std::function<bool(State*)> test;
  Pool* p;
  
  test = [=](State* s) -> bool {return s->status == 0;};
  p = device_task_pools[0];
  std::cout << "calling checkin" << std::endl;
  p->checkin(&advance_states, test);  

  test = [=](State* s) -> bool {return s->status == 1;};  
  p = device_task_pools[1];
  p->checkin(&advance_states, test);    


  test = [=](State* s) -> bool {return s->status == 2;};
  p = device_task_pools[2];
//  p = host_task_pools[0];    
  p->checkin(&advance_states, test);    
  
  std::cout << "leaving advance ..." << std::endl;  
}


int main(int argc, char* argv[]) {

  State* state = NULL;
  int size = 1000;

  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaDeviceSynchronize();
  std::cout << "device status before Graph: " << cudaGetErrorString(cuda_status) << std::endl;

  cuda_status = cudaMallocManaged(&state, sizeof(State)*size);
  assert(cuda_status == cudaSuccess);

  // create a Graph with 1 host task pool and 2 device task pools
  Graph* task_graph = new Graph(size, 0, 3);
  //Graph* task_graph = new Graph(size, 1, 2);  

  cuda_status = cudaDeviceSynchronize();
  std::cout << "device status after Graph: " << cudaGetErrorString(cuda_status) << std::endl;

  for (int i = 0; i < size; i++) {
    state[i].x = 2.0;
    if (i > 200 && i < 700)
      state[i].status = 1;
    else if (i <= 200)
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

  if (xsum == 4096*size) std::cout << "SUCCESS!" << std::endl;
  else std::cout << "ERROR!" << std::endl;
  
  return 0;
}
