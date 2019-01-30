#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H

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
#endif
