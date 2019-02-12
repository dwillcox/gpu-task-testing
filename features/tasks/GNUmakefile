all:
	nvcc -std=c++14 -gencode=arch=compute_60,code=sm_60 -rdc=true -I/usr/local/cuda-10.0/include -lcudadevrt -o main.exe main.cu
#	nvcc -std=c++14 --expt-extended-lambda -I/usr/local/cuda-10.0/include -o main.exe main.cu

clean:
	rm *.exe *~ *.o
