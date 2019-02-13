#include "unified.h"

void* UnifiedMemoryClass::operator new(size_t size) {
    void* vp;
    cudaMallocManaged(&vp, size);
    cudaDeviceSynchronize();
    return vp;
}

void UnifiedMemoryClass::operator delete(void* vp) {
    cudaDeviceSynchronize();
    cudaFree(vp);
}
