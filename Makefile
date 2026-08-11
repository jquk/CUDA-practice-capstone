# MNIST Data Setup Makefile

# Define content path
DATA_RELATIVE = data/
CXXFLAGS += -DDATA_DIR=\"$(DATA_RELATIVE)\"
CURRENT_DIR := $(shell pwd)
# MNIST_URL := http://yann.lecun.com/exdb/mnist
MNIST_URL := https://storage.googleapis.com/tensorflow/tf-keras-datasets
DATA_DIR := $(CURRENT_DIR)/$(DATA_RELATIVE)
BUILD_DIR := $(CURRENT_DIR)/build/

.PHONY: info
info:
	@echo "The dataset is located at the directory \`$(DATA_DIR)\`"
	@echo "which contains the following dataset items:"
	@ls -la $(DATA_DIR)/
	@echo "And the build directory contains:"
	@tree $(BUILD_DIR)/

.PHONY: all
all: download-mnist clean build-all

.PHONY: download-mnist
# Download MNIST digits dataset and move it to the expected path
download-mnist:
	mkdir -p $(DATA_DIR)
	wget $(MNIST_URL)/train-images-idx3-ubyte.gz -P $(DATA_DIR)/
	wget $(MNIST_URL)/train-labels-idx1-ubyte.gz -P $(DATA_DIR)/
	wget $(MNIST_URL)/t10k-images-idx3-ubyte.gz -P $(DATA_DIR)/
	wget $(MNIST_URL)/t10k-labels-idx1-ubyte.gz -P $(DATA_DIR)/
	$(MAKE) extract-mnist-dataset

.PHONY: extract-mnist-dataset
extract-mnist-dataset:
	gunzip -f $(DATA_DIR)/*.gz

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf build/*

clean-dataset:
	rm -rf $(DATA_DIR)

clean-all: clean clean-dataset

# Compile both programs
.PHONY: build-all
build-all: build-for-cpu build-for-gpu
#	$(MAKE) mnist_titest_on_cpu
#	$(MAKE) mnist_titest_on_gpu

# Compile the program that will perform training inference and tests on the CPU
.PHONY: build-for-cpu
build-for-cpu: $(SOURCES)
	mkdir -p build/bin
# 	$(CXX) $(CXXFLAGS) -o $@ $^
	$(CXX) $(CXXFLAGS) src/mnist_titest_on_cpu.cpp lib/helpers.cpp lib/nn_cpu.cpp -o build/bin/mnist_titest_on_cpu -Wall -Wextra -std=c++11

# Compile the program that will perform training inference and tests on the GPUs
.PHONY: build-for-gpu
build-for-gpu: $(SOURCES)
	mkdir -p build/bin
	nvcc $(CXXFLAGS) src/mnist_titest_on_gpu.cpp lib/nn_gpu.cpp lib/gpu_helpers.cu lib/helpers.cpp -o build/bin/mnist_titest_on_gpu -std=c++11 -lcudnn -lcublas

.PHONY: run-all
run-all:
	@echo "Running MNIST training, inference, and test on CPU, passing example program parameter values 5 128 0.01"
	@echo "Running MNIST training, inference, and test on GPU, passing example program parameter values 5 128 0.01"
	./build/bin/mnist_titest_on_gpu $(ARGS)
	./build/bin/mnist_titest_on_cpu $(ARGS)

.PHONY: run-on-cpu
run-on-cpu:
	@echo "Running MNIST training, inference, and test on CPU..."
	./build/bin/mnist_titest_on_cpu $(ARGS)

.PHONY: run-on-gpu
run-on-gpu:
	@echo "Running MNIST training, inference, and test on GPU..."
	./build/bin/mnist_titest_on_gpu $(ARGS)


# ============================================
# QUANTIZATION RECIPES
# For Quantization and Microcontroller Targets
# ============================================

# Paths
QUANT_DIR = src/quantization
MCU_DIR = $(QUANT_DIR)/microcontroller
ARDUINO_UNO_DIR = $(MCU_DIR)/arduino-uno
BUILD_DIR = build

# Build utility objects
$(BUILD_DIR)/quantize_utils.o: $(QUANT_DIR)/quantize_utils.cpp $(QUANT_DIR)/quantize_utils.h
	@mkdir -p $(BUILD_DIR)
	g++ -c -O3 -std=c++11 $< -o $@

# Train tiny model
train-tiny: $(BUILD_DIR)/train_tiny_model
	./$(BUILD_DIR)/train_tiny_model

$(BUILD_DIR)/train_tiny_model: $(QUANT_DIR)/train_tiny_model.cpp $(BUILD_DIR)/quantize_utils.o
	g++ -O3 -std=c++17 $^ -o $@

help:
	@echo "TRAINING & INFERENCE IN CPU AND GPU:"
	@echo "  make info                     - Show where the dataset and build directories are and their contents."
	@echo "  make all                      - Download MNIST dataset, clean and build."
	@echo "  make download-mnist           - Download MNIST's handwritten digits dataset."
	@echo "  make extract-mnist-dataset    - Extract the MNIST dataset."
	@echo "  make clean                    - Remove contents in 'build/' directory."
	@echo "  make clean-dataset            - Remove the MNIST dataset."
	@echo "  make clean-all                - Remove contents in 'build/' dir and MNIST dataset."
	@echo "  make build-all                - Build neural network and compile for both the CPU and GPU."
	@echo "  make build-for-cpu            - Build for CPU (uses specific headers)."
	@echo "  make build-for-gpu            - Build for GPU (uses NVIDIA's CUDA libraries and specific headers)."
	@echo "  make run-all                  - Run both the CPU version and GPU version programs to compare results,"
	@echo "                                  pass parameters as \`make run-all ARGS=\"<epochs>, <hidden_layers_size>, <learning_rate>\"\`."
	@echo "  make run-on-cpu               - Run only the CPU oriented program,"
	@echo "                                  pass parameters as in \`make run-all\`."
	@echo "  make run-on-gpu               - Run only the GPU oriented program,"
	@echo "                                  pass parameters as in \`make run-all\`."
	@echo "QUANTIZATION & MICROCONTROLLER TARGETS:"
	@echo "  make train-tiny               - Train 7x7→8→10 model"
