#ifndef TASK_H
#define TASK_H
#include "lock.h"
#include "generic_function.h"
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>

class Task : public UnifiedMemoryClass {
public:
  GenericFunction* gfun;
  void* state;
  __host__ __device__ Task(GenericFunction* gf = NULL, void* args = NULL) : gfun(gf), state(args) {}
  __host__ __device__ void execute() {
    gfun->execute(state);
  }
};
#endif
