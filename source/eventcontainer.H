#ifndef EVENTCONTAINER_H
#define EVENTCONTAINER_H
#include <cuda.h>
#include <cuda_runtime.h>

class EventContainer {
    cudaEvent_t event;
    cudaError_t event_status;
public:
    EventContainer() {
        cudaEventCreate(&event);
    }

    ~EventContainer() {}

    bool query_cuda_event() {
        event_status = cudaEventQuery(event);
        return (event_status == cudaSuccess);
    }

    void stream_insert_event(cudaStream_t& stream) {
        cudaEventRecord(event, stream);
    }
};
#endif
