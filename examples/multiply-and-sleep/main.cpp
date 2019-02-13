#include <iostream>
#include <cassert>
#include "graph.h"
#include "state.h"

void square_host(void* xv) {
  State* state = static_cast<State*>(xv);
  state->x = (state->x) * (state->x);
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
