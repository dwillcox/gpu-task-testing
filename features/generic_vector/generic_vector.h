#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "unified.h"

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

  __host__ __device__ T& operator[] (unsigned int i) {
    #ifdef __CUDA_ARCH__
    return vdevice[i];
    #else
    return vhost[i];
    #endif
  }

  __host__ __device__ T* get_pointer_to (unsigned int i) {
    #ifdef __CUDA_ARCH__
    return &device_data_ptr[i];
    #else
    T* hdata = thrust::raw_pointer_cast(vhost.data());
    return &hdata[i];
    #endif
  }

  void sync_from_device() {
    vhost = vdevice;
    update_size();
  }

  void sync_to_device() {
    vdevice = vhost;
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

  __host__ __device__ size_t size() {
    return filled_size;
  }
};
#endif
