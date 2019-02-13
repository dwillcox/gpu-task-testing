#include <iostream>
#include <cassert>
#include <functional>
#include "graph.h"
#include "state.h"

#define USE_HOST_BATCHED_POOL 1

void square_host(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
}

std::function<int (State*)> create_state_pool_map() {
    // Return value is the global pool index spanning host and device pools.
    // Pools are indexed from 0 to N, with host pools first and then device pools next.
    return [](State* s) -> int {
        // This implementation is meant for num_host_pools + num_device_pools = 3;
        // This choice of pool index will put status=2 states into a host pool
        // if num_host_pools = 1 and into a device pool if num_host_pools = 0.
        if (s->status == 1) return 2;
        else if (s->status == 0) return 1;
        else if (s->status == 2) return 0;
        else return -1; // there is no pool -1 so these states won't go into any pool.
    };
}

int main(int argc, char* argv[]) {

  State* state = NULL;
  int size = 1000;
  size_t num_host_pools = 0;
  size_t num_device_pools = 3;

  cudaError_t cuda_status = cudaSuccess;
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);  

  cuda_status = cudaMallocManaged(&state, sizeof(State)*size);
  assert(cuda_status == cudaSuccess);

  if (USE_HOST_BATCHED_POOL) {
      num_host_pools = 1;
      num_device_pools = 2;
  }

  Graph* task_graph = new Graph(size, num_host_pools, num_device_pools);

  task_graph->set_state_pool_map_function(create_state_pool_map());

  auto check_state_completed = [](State* s) -> bool {return s->status == 3;};
  task_graph->set_state_completed_function(check_state_completed);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);  

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
