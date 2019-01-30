#ifndef UNIFIED_H
#define UNIFIED_H
class UnifiedMemoryClass
{
public:
  void* operator new(size_t size) {
    void* vp;
    cudaMallocManaged(&vp, size);
    cudaDeviceSynchronize();
    return vp;
  }

  void operator delete(void* vp) {
    cudaDeviceSynchronize();
    cudaFree(vp);
  }
};
#endif
