# Pascal Unified Memory Benchmark Makefile
# Compile for Pascal SM 6.1 (GTX 1080)

NVCC = nvcc
ARCH = -arch=sm_61
OPTS = -O3

pascal: pascal.cu
	$(NVCC) $(ARCH) $(OPTS) pascal.cu -o pascal

clean:
	rm -f pascal

.PHONY: clean
