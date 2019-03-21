#ifndef STREAMCONTAINER_H
#define STREAMCONTAINER_H
#include <cuda.h>
#include <cuda_runtime.h>

class StreamContainer {
    cudaStream_t* stream;
public:
    StreamContainer() {
        cudaError_t cuda_status = cudaStreamCreate(stream);
        assert(cuda_status == cudaSuccess);
    }

    ~StreamContainer() {
        cudaError_t cuda_status = cudaStreamDestroy(*stream);
        assert(cuda_status == cudaSuccess);
    }

    cudaStream_t& get_stream() {
        return *stream;
    }
};
#endif
