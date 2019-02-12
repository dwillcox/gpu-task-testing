#ifndef LOCK_H
#define LOCK_H
#include <cassert>
#include "unified.h"

class HostDeviceLock : public UnifiedMemoryClass {
  int* mutex;
  cudaStream_t lockStream;
public:
  HostDeviceLock();

  __host__ __device__ void lock();

  __host__ __device__ void unlock();
};
#endif
