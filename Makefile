# Compile the program that will perform training inference and tests on the CPU
titest:
	g++ mnist_titest.cpp -o mnist_titest -Wall -Wextra -std=c++11

# Compile the program that will perform training inference and tests on the GPUs
titestongpu:
	nvcc mnist_titest_on_gpu.cpp -o mnist_titest_on_gpu -std=c++11 -lcudnn -lcublas
