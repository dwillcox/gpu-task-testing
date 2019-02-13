#include "streamcontainer.H"

__host__ __device__ void StreamContainer::set_stream(cudaStream_t* cuda_stream_ptr) {
    stream_ptr = cuda_stream_ptr;
}

__host__ __device__ cudaStream_t& StreamContainer::get_stream() {
    return *stream_ptr;
}
