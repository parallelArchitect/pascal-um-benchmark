/*
 * Pascal Unified Memory Benchmark — CUDA Kernel
 *
 * Minimal vector kernel used to measure Unified Memory behavior on Pascal GPUs.
 * Designed to expose demand-paged cudaMallocManaged migration versus
 * cudaMemPrefetchAsync DRAM-resident execution.
 *
 * Reference:
 *   https://stackoverflow.com/questions/39782746
 *
 * Repository:
 *   https://github.com/parallelArchitect/pascal-um-benchmark
 *
 * Author: Joe McLaren — Human–AI Collaborative Engineering
 * License: MIT
 * Version: 2.4.0
 *
 * Tested On:
 *   - GPU: NVIDIA GeForce GTX 1080 (8 GB GDDR5X, SM 6.1)
 *   - Driver: 535.274.02
 *   - CUDA Toolkit: 12.0
 *   - Compiler: nvcc 12.0 (V12.0.140)
 *   - OS: Ubuntu 24.04
 */


#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../pcie/pcie_bandwidth.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void vec_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

double measure_bandwidth(float *a, float *b, float *c, size_t n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    vec_add<<<grid_size, block_size>>>(a, b, c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    size_t bytes = n * sizeof(float) * 3;
    double bandwidth = (bytes / ms) / 1e6;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return bandwidth;
}

void get_timestamp(char *buf, size_t bufsize, bool filename_safe) {
    time_t now;
    struct tm *tm_info;
    time(&now);
    tm_info = localtime(&now);
    
    if (filename_safe) {
        // For filenames: 2025-10-31_04-15-30_PM_EST
        strftime(buf, bufsize, "%Y-%m-%d_%I-%M-%S_%p_%Z", tm_info);
    } else {
        // For display: 2025-10-31 04:15:30 PM EST
        strftime(buf, bufsize, "%Y-%m-%d %I:%M:%S %p %Z", tm_info);
    }
}

void create_results_dir(const char *subdir) {
    char path[512];
    
    // Create results/ if not exists
    mkdir("../results", 0755);
    
    // Create results/subdir/ if not exists
    snprintf(path, sizeof(path), "../results/%s", subdir);
    mkdir(path, 0755);
}

void save_log(double bw_naive, double bw_prefetch, const char *subdir) {
    create_results_dir(subdir);
    
    char timestamp[128];
    char filename[256];
    get_timestamp(timestamp, sizeof(timestamp), true);
    
    snprintf(filename, sizeof(filename), "../results/%s/run_%s.log", subdir, timestamp);
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Warning: Could not create log file: %s\n", filename);
        return;
    }
    
    char display_time[128];
    get_timestamp(display_time, sizeof(display_time), false);
    
    fprintf(f, "═══════════════════════════════════════════════════════════════════════\n");
    fprintf(f, "  Pascal UM Benchmark - Results Log\n");
    fprintf(f, "═══════════════════════════════════════════════════════════════════════\n");
    fprintf(f, "Timestamp: %s\n", display_time);
    fprintf(f, "\n");
    fprintf(f, "Results:\n");
    fprintf(f, "  Naive UM:      %.1f GB/s\n", bw_naive);
    fprintf(f, "  With Prefetch: %.1f GB/s\n", bw_prefetch);
    fprintf(f, "  Speedup:       %.1fx\n", bw_prefetch / bw_naive);
    fprintf(f, "═══════════════════════════════════════════════════════════════════════\n");
    
    fclose(f);
    
    printf("Log saved: %s\n", filename);
}

void output_json(double bw_naive, double bw_prefetch) {
    char timestamp[128];
    get_timestamp(timestamp, sizeof(timestamp), false);
    
    printf("{\n");
    printf("  \"timestamp\": \"%s\",\n", timestamp);
    printf("  \"tool\": \"pascal\",\n");
    printf("  \"gpu\": \"GTX 1080\",\n");
    printf("  \"tests\": [\n");
    printf("    {\"name\": \"naive\", \"bandwidth_gbs\": %.1f},\n", bw_naive);
    printf("    {\"name\": \"prefetch\", \"bandwidth_gbs\": %.1f}\n", bw_prefetch);
    printf("  ],\n");
    printf("  \"speedup\": %.1f\n", bw_prefetch / bw_naive);
    printf("}\n");
}

int main(int argc, char **argv) {
    // Parse flags
    bool pcie_test = false;
    bool log_results = false;
    bool json_output = false;
    bool quiet_mode = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pcie") == 0) {
            pcie_test = true;
        } else if (strcmp(argv[i], "--log") == 0) {
            log_results = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet_mode = true;
        }
    }
    
    size_t n = 256 * 1024 * 1024;
    size_t bytes = n * sizeof(float);
    
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    // Naive
    double bw_naive = measure_bandwidth(a, b, c, n);
    
    // Prefetch
    CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double bw_prefetch = measure_bandwidth(a, b, c, n);
    
    // Output based on mode
    if (json_output) {
        output_json(bw_naive, bw_prefetch);
    } else if (quiet_mode) {
        printf("%.1f,%.1f\n", bw_naive, bw_prefetch);
    } else {
        printf("Naive:      %.1f GB/s\n", bw_naive);
        printf("Prefetch:   %.1f GB/s\n", bw_prefetch);
    }
    
    // Save log if requested
    if (log_results) {
        save_log(bw_naive, bw_prefetch, "pascal");
    }
    
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    
    // PCIe bandwidth test (if --pcie flag provided)
    if (pcie_test && !json_output && !quiet_mode) {
        PCIeBandwidthResult pcie = measure_pcie_bandwidth();
        print_pcie_results(&pcie);
        analyze_page_faults(bw_naive, bw_prefetch, &pcie);
    }
    
    return 0;
}
