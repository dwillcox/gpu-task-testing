#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template<class T> __global__
void DeviceVector_push_back(void*, T);

template<class T> __global__
void DeviceVector_resize(void*, size_t);

template<class T> __global__
void DeviceVector_size(void*, size_t*);

template<class T> class DeviceVector {
  T* contents;
  T* end_filled_ptr;
  size_t allocated_size;

  __device__ void lengthen() {
    allocated_size *= 2;
    size_t filled_size = end_filled_ptr - contents;
    T* new_contents = new T[allocated_size];
    memcpy(new_contents, contents, filled_size*sizeof(T));
    delete contents;
    contents = new_contents;
    end_filled_ptr = contents + filled_size;
  }
public:
  __device__ DeviceVector() : allocated_size(16) {
    contents = new T[allocated_size];
    end_filled_ptr = contents;
  }
  
  __device__ DeviceVector(size_t create_size) : allocated_size(create_size) {
    contents = new T[allocated_size];
    end_filled_ptr = contents;
  }
  
  __device__ ~DeviceVector() {
    delete[] contents;
  }

  __device__ T& operator[] (unsigned int i) {
    return *(contents + i);
  }

  void copy_host_to_device(std::vector<T>* host_vector) {
    T* host_data = host_vector->data();
    size_t hsize = host_vector->size();
    DeviceVector_resize<T><<<1,1>>>(this, hsize);
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
    cuda_status = cudaMemcpy(contents, host_data, hsize*sizeof(T), cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);
  }

  void copy_device_to_host(std::vector<T>* host_vector) {
    cudaError_t cuda_status;
    
    T* host_data = host_vector->data();
    size_t* dsize_p;

    cuda_status = cudaMallocManaged(&dsize_p, sizeof(size_t));
    assert(cuda_status == cudaSuccess);
    
    DeviceVector_size<T><<<1,1>>>(this, dsize_p);
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    host_vector->resize(*dsize_p);
    
    cuda_status = cudaMemcpy(host_data, contents, (*dsize_p)*sizeof(T), cudaMemcpyDeviceToHost);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaFree(dsize_p);
    assert(cuda_status == cudaSuccess);
  }

  __device__ T* begin() {
    return contents;
  }

  __device__ T* end() {
    return end_filled_ptr;
  }

  __device__ size_t size() {
    return end_filled_ptr - contents;
  }

  __device__ void push_back(T element) {
    size_t filled_size = end_filled_ptr - contents + 1;
    if (filled_size > allocated_size) lengthen();
    new (end_filled_ptr) T(element);
    end_filled_ptr++;
  }

  __device__ void resize(size_t new_size) {
    size_t filled_size = end_filled_ptr - contents;
    if (new_size == filled_size) return;
    T* new_contents = new T[new_size];
    size_t copy_size = min(filled_size, new_size);
    memcpy(new_contents, contents, copy_size*sizeof(T));
    contents = new_contents;
    end_filled_ptr = contents + copy_size;
  }
};

template<class T> __global__
void DeviceVector_resize(void* vdvec, size_t new_size) {
  DeviceVector<T>* dvec = static_cast<DeviceVector<T>*>(vdvec);
  dvec->resize(new_size);
}

template<class T> __global__
void DeviceVector_size(void* vdvec, size_t* vsize) {
  DeviceVector<T>* dvec = static_cast<DeviceVector<T>*>(vdvec);  
  *vsize = dvec->size();
}

template<class T> __global__
void DeviceVector_push_back(void* vdvec, T element) {
  DeviceVector<T>* dvec = static_cast<DeviceVector<T>*>(vdvec);  
  dvec->push_back(element);
}

#endif
