# Compile the program that will perform training inference and tests on the CPU
titest:
	mkdir -p bin
	g++ mnist_titest.cpp -o bin/mnist_titest -Wall -Wextra -std=c++11

# Compile the program that will perform training inference and tests on the GPUs
titestongpu:
	mkdir -p bin
	nvcc mnist_titest_on_gpu.cpp -o bin/mnist_titest_on_gpu -std=c++11 -lcudnn -lcublas
