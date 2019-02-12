all:
	nvcc -std=c++14 -gencode=arch=compute_70,code=sm_70 -rdc=true -I/usr/local/cuda-10.0/include -lcudadevrt -o main.exe main.cu

clean:
	rm *.exe *~ *.o
