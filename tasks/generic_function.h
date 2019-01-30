#ifndef GENERIC_FUNCTION_H
#define GENERIC_FUNCTION_H

typedef void (* execute_task_t)(void *);

class GenericFunction : public UnifiedMemoryClass {
  execute_task_t* device_function;
  execute_task_t* host_function;
public:
  GenericFunction(execute_task_t* host_ptr = nullptr, execute_task_t* device_ptr = nullptr) {
    host_function = host_ptr;
    device_function = device_ptr;
  }

  __host__ __device__ void execute(void* args) {
    #ifdef __CUDA_ARCH__
    (*device_function)(args);
    #else
    (*host_function)(args);
    #endif
  }

  __host__ __device__ bool supports_host() {
    return host_function;
  }

  __host__ __device__ bool supports_device() {
    return device_function;
  }
};

#endif
