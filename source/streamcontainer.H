#ifndef STREAMCONTAINER_H
#define STREAMCONTAINER_H
#include <cuda.h>
#include <cuda_runtime.h>

class StreamContainer {
    cudaStream_t* stream_ptr;
public:
    __host__ __device__ void set_stream(cudaStream_t*);

    __host__ __device__ cudaStream_t& get_stream();
};
#endif
