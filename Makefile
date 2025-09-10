# MNIST Data Setup Makefile

# Define content path
CONTENT_RELATIVE = content/
CXXFLAGS += -DDATA_DIR=\"$(CONTENT_RELATIVE)\"
CURRENT_DIR := $(shell pwd)
CONTENT_DIR := $(CURRENT_DIR)/$(CONTENT_RELATIVE)
# MNIST_URL := http://yann.lecun.com/exdb/mnist
MNIST_URL := https://storage.googleapis.com/tensorflow/tf-keras-datasets

.PHONY: info
info:
	@echo "Content directory: $(CONTENT_DIR)"
	@ls -la $(CONTENT_DIR)/

.PHONY: all
all: download-mnist extract clean build run

.PHONY: download-mnist
# Download MNIST digits dataset and move it to the expected path
download-mnist:
	mkdir -p $(CONTENT_DIR)
	wget $(MNIST_URL)/train-images-idx3-ubyte.gz -P $(CONTENT_DIR)/
	wget $(MNIST_URL)/train-labels-idx1-ubyte.gz -P $(CONTENT_DIR)/
	wget $(MNIST_URL)/t10k-images-idx3-ubyte.gz -P $(CONTENT_DIR)/
	wget $(MNIST_URL)/t10k-labels-idx1-ubyte.gz -P $(CONTENT_DIR)/

.PHONY: extract
extract:
	gunzip -f $(CONTENT_DIR)/*.gz

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(CONTENT_DIR)
	rm bin/*

# Compile both programs
.PHONY: build-all
build-all: build-for-cpu build-for-gpu
#	$(MAKE) mnist_titest_on_cpu
#	$(MAKE) mnist_titest_on_gpu

# Compile the program that will perform training inference and tests on the CPU
.PHONY: build-for-cpu
build-for-cpu: $(SOURCES)
	mkdir -p bin
# 	$(CXX) $(CXXFLAGS) -o $@ $^
	$(CXX) $(CXXFLAGS) src/mnist_titest_on_cpu.cpp lib/helpers.cpp lib/nn_cpu.cpp -o bin/mnist_titest_on_cpu -Wall -Wextra -std=c++11

# Compile the program that will perform training inference and tests on the GPUs
.PHONY: build-for-gpu
build-for-gpu: $(SOURCES)
	mkdir -p bin
	nvcc $(CXXFLAGS) src/mnist_titest_on_gpu.cpp lib/nn_gpu.cpp lib/gpu_helpers.cu lib/helpers.cpp -o bin/mnist_titest_on_gpu -std=c++11 -lcudnn -lcublas

.PHONY: run-all
run-all: run-mnist_titest_on_cpu run-mnist_titest_on_gpu

.PHONY: run-on-cpu
run-on-cpu:
	@echo "Running MNIST training, inference, and test on CPU..."
	./bin/mnist_titest_on_cpu

.PHONY: run-on-gpu
run-on-gpu:
	@echo "Running MNIST training, inference, and test on GPU..."
	./bin/mnist_titest_on_gpu
