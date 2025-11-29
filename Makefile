# Pascal Unified Memory Benchmark — Build Script
#
# Builds the CUDA benchmark used to evaluate Unified Memory page-fault migration
# and cudaMemPrefetchAsync optimization on Pascal GPUs.
#
# Reference:
#   https://stackoverflow.com/questions/39782746
#
# Repository:
#   https://github.com/parallelArchitect/pascal-um-benchmark
#
# Author: Joe McLaren — Human–AI Collaborative Engineering
# License: MIT
# SPDX-License-Identifier: MIT
# Version: 2.4.0
#
# Tested On:
#   - GPU: NVIDIA GeForce GTX 1080 (SM 6.1)
#   - Driver: 535.274.02
#   - CUDA Toolkit: 12.0
#   - OS: Ubuntu 24.04

NVCC = nvcc
ARCH = -arch=sm_61
OPTS = -O3

pascal: pascal.cu
	$(NVCC) $(ARCH) $(OPTS) pascal.cu -o pascal

clean:
	rm -f pascal

.PHONY: clean
