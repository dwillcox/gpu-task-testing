#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "unified.h"
#include "streamcontainer.h"

template<class T> class GenericVector : public UnifiedMemoryClass {
  thrust::host_vector<T> vhost;
  thrust::device_vector<T> vdevice;
  size_t filled_size;
  T* device_data_ptr;
public:
  GenericVector(size_t create_size = 0) {
    vhost.resize(create_size);
    vdevice.resize(create_size);
    update_size();
    device_data_ptr = thrust::raw_pointer_cast(vdevice.data());
  }
  
  ~GenericVector() {
  }

  __host__ __device__ T* begin() {
    #ifdef __CUDA_ARCH__
    return device_data_ptr;
    #else
    return thrust::raw_pointer_cast(vhost.data());
    #endif
  }

  __host__ __device__ T* end() {
    #ifdef __CUDA_ARCH__
    return device_data_ptr + filled_size;
    #else
    return thrust::raw_pointer_cast(vhost.data()) + filled_size;
    #endif
  }

  __host__ __device__ T& operator[] (unsigned int i) {
    T* tptr = get_pointer_to(i);
    return *tptr;
  }

  __host__ T* get_host_pointer_to (unsigned int i) {
    T* hdata = thrust::raw_pointer_cast(vhost.data());
    return &hdata[i];
  }

  __host__ __device__ T* get_device_pointer_to (unsigned int i) {
    return &device_data_ptr[i];
  }

  __host__ __device__ T* get_pointer_to (unsigned int i) {
    #ifdef __CUDA_ARCH__
    return get_device_pointer_to(i);
    #else
    return get_host_pointer_to(i);
    #endif
  }

  void sync_from_device() {
    cudaEvent_t thrust_finished;
    cudaEventCreate(&thrust_finished);

    vhost = vdevice;

    cudaEventRecord(thrust_finished);
    cudaEventSynchronize(thrust_finished);

    update_size();
  }

  void sync_to_device() {
    cudaEvent_t thrust_finished;
    cudaEventCreate(&thrust_finished);
    
    vdevice = vhost;

    cudaEventRecord(thrust_finished);
    cudaEventSynchronize(thrust_finished);
    
    device_data_ptr = thrust::raw_pointer_cast(vdevice.data());
    update_size();
  }

  void update_size() {
    filled_size = vhost.size();
  }

  void push_back(T element) {
    vhost.push_back(element);
    update_size();
  }

  void resize(size_t new_size) {
    vhost.resize(new_size);
    sync_to_device();
  }

  __host__ __device__ size_t size() {
    return filled_size;
  }
};
#endif
