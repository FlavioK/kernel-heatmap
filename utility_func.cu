#include "utility_func.cuh"
#include "utility_host.hpp"
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Prints a message and returns zero if the given value is not cudaSuccess
#define CheckCUDAError(val) (InternalCheckCUDAError((val), #val, __FILE__, __LINE__))

// Called internally by CheckCUDAError
static inline int InternalCheckCUDAError(cudaError_t result, const char *fn,
        const char *file, int line) {
    if (result == cudaSuccess) return 0;
    printf("CUDA error %d in %s, line %d (%s): %s\n", (int) result, file, line,
            fn, cudaGetErrorString(result));
    return -1;
}

static __global__ void getStartTimeInternal(uint64_t *targetStartTime) {
    if(threadIdx.x == 0){
        *targetStartTime = getTime() + START_TIME_OFFSET_NS;
    }
    __syncthreads();
}

void getStartTime(uint64_t *targetStartTime){
        getStartTimeInternal<<<1,1>>>(targetStartTime);

        if (CheckCUDAError(cudaDeviceSynchronize())) perror("Could not synchronize device\n");
}
