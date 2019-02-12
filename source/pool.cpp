#include "pool.h"

__global__ void pool_kernel(Pool* pool) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t size = pool->checked_out_tasks.size();
  if (tid < size) {
      pool->checked_out_tasks[tid]->advance();
  }
}
