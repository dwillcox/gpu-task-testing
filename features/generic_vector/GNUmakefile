CUDA_ARCH ?= 70

all:
	nvcc -std=c++14 -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -rdc=true -I/usr/local/cuda-10.0/include -lcudadevrt -o main.exe main.cu

clean:
	rm *.exe *~ *.o
