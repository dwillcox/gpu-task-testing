CUDA_ARCH ?= 70

all:
	g++ -std=c++14 -o main.gnu.exe main.cpp -lpthread
	nvcc -std=c++14 -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -rdc=true -I/usr/local/cuda-10.0/include -lcudadevrt -o main.cuda.exe -x cu main.cpp -lpthread

clean:
	rm *.exe *~ *.o
