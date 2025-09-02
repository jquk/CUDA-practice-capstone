# Download MNIST dataset and move it to the expected path
download:
	wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
	wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
	wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
	wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz

	gunzip train-images-idx3-ubyte.gz
	gunzip train-labels-idx1-ubyte.gz
	gunzip t10k-images-idx3-ubyte.gz
	gunzip t10k-labels-idx1-ubyte.gz

	mkdir -p content
	mv content /

	mv train-images-idx3-ubyte.gz /content/
	mv train-labels-idx1-ubyte.gz /content/
	mv t10k-images-idx3-ubyte.gz /content/
	mv t10k-labels-idx1-ubyte.gz /content/

# Compile the program that will perform training inference and tests on the CPU
titest:
	mkdir -p bin
	g++ mnist_titest.cpp -o bin/mnist_titest -Wall -Wextra -std=c++11

# Compile the program that will perform training inference and tests on the GPUs
titestongpu:
	mkdir -p bin
	nvcc mnist_titest_on_gpu.cpp -o bin/mnist_titest_on_gpu -std=c++11 -lcudnn -lcublas
