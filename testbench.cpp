#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "testbench.h"
#include "complexmat.cuh"
#include "utility_host.hpp"

// Maximum numbre of used blocks in launched kernels.
// This number is used to determine how many time slots have to be allocated.
#define MAX_NOF_BLOCKS  (4)

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

#define DEVICE_NUMBER (0)

static kernelFunc assignKernel(kernel_type_t kernel){
    switch (kernel){
    case KER_SQR_NORM:
        return &sqr_norm;
    case KER_CONJ:
        return &conj;
    case KER_MULT:
        return &same_num_channels_mul;
    case KER_GAUSS:
        return &GaussianCorrelation;
#if 0
    case KER_SQR_MAG:
        return &sqr_mag_kernel;
    case KER_CONJ:
        return &conj_kernel;
    case KER_SUM_CHANN:
        return &sum_channels;
    case KER_MULT:
        return &same_num_channels_mul_kernel;
    case KER_DIV:
        return &same_num_channels_div_kernel;
    case KER_ADD:
        return &same_num_channels_add_kernel;
    case KER_MUL_C:
        return &constant_mul_kernel;
    case KER_ADD_C:
        return &constant_add_kernel;
    case KER_ELEM_MUL:
        return &one_channel_mul_kernel;
#endif
    default:
        return NULL;
    }
}

int initializeTest(param_t *params){

    // Set CUDA device
    if (CheckCUDAError(cudaSetDevice(DEVICE_NUMBER))) {
        return EXIT_FAILURE;
    }

    // Allocate kerneldata
    kernel_data *kernelData = NULL;
    kernelData = (kernel_data*)malloc(params->nofKernels*sizeof(kernel_data));
    if (!kernelData) {
        perror("Failed allocating kerneldata: ");
        return  -1;
    }
    memset(kernelData, 0, params->nofKernels*sizeof(kernel_data));

    // Allocate device startTime
    if (CheckCUDAError(cudaMalloc(&params->targetStartTime, \
                    sizeof(uint64_t)))) return -1;

    // Fill kernel_data structures
    for(int i = 0; i < params->nofKernels; i++){

        if (CheckCUDAError(cudaHostAlloc(&kernelData[i].interfering, \
                        sizeof(int), cudaHostAllocMapped))) return -1;
        // Assign kernel function
        if(i == 0){
            kernelData[i].kernelFunc = assignKernel(params->kernelUT);
            *kernelData[i].interfering = 0;
        } else {
            kernelData[i].kernelFunc = assignKernel(params->kernelInter);
            *kernelData[i].interfering = 1;
        }

        // Create stream
        cudaStreamCreate(&kernelData[i].stream);
        // Allocate data
#ifdef USE_CUDA_MEMCPY
        if (CheckCUDAError(cudaMalloc(&kernelData[i].dummyData1, \
                        params->data_size*sizeof(float)))) return -1;    
        if (CheckCUDAError(cudaMalloc(&kernelData[i].dummyData2, \
                        params->data_size*sizeof(float)))) return -1;    
        if (CheckCUDAError(cudaMalloc(&kernelData[i].dummyData3, \
                        params->data_size*sizeof(float)))) return -1;    
#else
        if (CheckCUDAError(cudaHostAlloc(&kernelData[i].dummyData1, \
                        params->data_size*sizeof(float), cudaHostAllocMapped))) return -1;    
        if (CheckCUDAError(cudaHostAlloc(&kernelData[i].dummyData2, \
                        params->data_size*sizeof(float), cudaHostAllocMapped))) return -1;    
        if (CheckCUDAError(cudaHostAlloc(&kernelData[i].dummyData3, \
                        params->data_size*sizeof(float), cudaHostAllocMapped))) return -1;    
#endif

        // Allocate time array
        if (CheckCUDAError(cudaMalloc(&kernelData[i].targetTimes, \
                        MAX_NOF_BLOCKS * 2 * sizeof(uint64_t)))) return -1;

        // Allocate smid space
        if (CheckCUDAError(cudaMalloc(&kernelData[i].smid, \
                        MAX_NOF_BLOCKS * sizeof(unsigned int)))) return -1;

        kernelData[i].startTime = params->targetStartTime;
        kernelData[i].dataSize = params->data_size;
    }

#if 0 // Will be used after we have ran the test
    //allocate host times
    params->blockTimes = NULL;
    params->blockTimes = (uint64_t *) malloc(params->nof_repetitions*2*params->nofBlocks*params->nofKernels*sizeof(uint64_t));
    if (!params->blockTimes) {
        perror("Failed allocating hostTimes buffer: ");
        return  -1;
    }
    memset(params->blockTimes,0, params->nof_repetitions*2*params->nofBlocks*params->nofKernels*sizeof(uint64_t));

    // allocate host smid buffer
    params->smid = NULL;
    params->smid = (unsigned int *) malloc(params->nofKernels*params->nofBlocks*params->nof_repetitions*sizeof(unsigned int));
    if (!params->smid) {
        perror("Failed allocating smid buffer: ");
        return  -1;
    }
    memset(params->smid,0 , params->nofKernels*params->nofBlocks*params->nof_repetitions*sizeof(float));
#endif

    // Assigne kernel data
    params->kernelData = (void*) kernelData;
    return 0;
}

int runTest(param_t *params){

    kernel_data *kernelData = (kernel_data*)params->kernelData;

    for(int rep = -1; rep < params->nof_repetitions; rep++){

        // Get measurement startTime
        getStartTime(params->targetStartTime);

        // Launch all kernel
        for(int kernel = params->nofKernels-1; kernel >= 0; kernel--){
            // Launch kernel
            kernelData[kernel].kernelFunc(&kernelData[kernel]);
        }
        if (CheckCUDAError(cudaStreamSynchronize(kernelData[0].stream))) perror("Problem with stream sync");

        for(int kernel = 1; kernel < params->nofKernels; kernel++){
            // Disable interfering kernels
            *kernelData[kernel].interfering = 0;
        }
        if (CheckCUDAError(cudaDeviceSynchronize())) perror("Problem with device sync");
        for(int kernel = 1; kernel < params->nofKernels; kernel++){
            // Renable interfering kernels
            *kernelData[kernel].interfering = 1;
        }

        if(rep>=0){
            // Store data if no warm up iteration
            if (CheckCUDAError(cudaMemcpy(&params->blockTimes[2*kernelData[0].nofBlocks*rep], \
                                          kernelData[0].targetTimes, \
                                          2*kernelData[0].nofBlocks*sizeof(uint64_t), \
                                          cudaMemcpyDeviceToHost))) return -1;

            // Copyback smid's
            if (CheckCUDAError(cudaMemcpy(&params->smid[kernelData[0].nofBlocks*rep], \
                                          kernelData[0].smid, \
                                          kernelData[0].nofBlocks*sizeof(unsigned int), \
                                          cudaMemcpyDeviceToHost))) return -1;
            uint64_t minStart=UINT64_MAX, maxEnd=0;
            uint64_t *currVals = &params->blockTimes[2*kernelData[0].nofBlocks*rep];
            for(uint i = 0 ; i<kernelData[0].nofBlocks;i++){
                if(currVals[i*kernelData[0].nofBlocks] < minStart){
                   minStart = currVals[i*kernelData[0].nofBlocks];
                }
                if(currVals[i*kernelData[0].nofBlocks + 1] > maxEnd){
                   maxEnd = currVals[i*kernelData[0].nofBlocks + 1];
                }
            }
            params->kernelDurations[rep] = maxEnd-minStart;

        } else{
            // Allocate host data storage after warm-up iteration
            //allocate host times
            params->blockTimes = NULL;
            size_t numberOfBlocksAllReps = params->nof_repetitions * kernelData[0].nofBlocks;
            params->blockTimes = (uint64_t *) malloc(2 * numberOfBlocksAllReps * sizeof(uint64_t));
            if (!params->blockTimes) {
                perror("Failed allocating hostTimes buffer: ");
                return  -1;
            }
            memset(params->blockTimes,0, 2 * numberOfBlocksAllReps * sizeof(uint64_t));

            // allocate host smid buffer
            params->smid = NULL;
            params->smid = (unsigned int *) malloc(numberOfBlocksAllReps * sizeof(unsigned int));
            if (!params->smid) {
                perror("Failed allocating smid buffer: ");
                return  -1;
            }
            memset(params->smid,0 , numberOfBlocksAllReps * sizeof(unsigned int));

            //Allocate kernel durations
            params->kernelDurations = NULL;
            params->kernelDurations = (uint64_t *) malloc(params->nof_repetitions * sizeof(uint64_t));
            if (!params->kernelDurations) {
                perror("Failed allocating kernelDurations buffer: ");
                return  -1;
            }
            memset(params->kernelDurations,0, params->nof_repetitions * sizeof(uint64_t));
        }
    }

    return 0;
}

int writeResults(param_t *params){
    kernel_data *kernelData = (kernel_data*)params->kernelData;
    if (fprintf(params->fd,"{\n") < 0 ) return -1;
    // Write device info
    cudaDeviceProp deviceProp;
    if (CheckCUDAError(cudaGetDeviceProperties(&deviceProp, DEVICE_NUMBER))) return -1;
    if (fprintf(params->fd,"\"clockRate\": \"%d\",\n", deviceProp.clockRate)  < 0 ) return -1;

    // Write header
    if (fprintf(params->fd,"\"kernel_ut\": \"%s\",\n", getKernelString(params->kernelUT))  < 0 ) return -1;
    if (fprintf(params->fd,"\"kernel_inter\": \"%s\",\n", getKernelString(params->kernelInter))  < 0 ) return -1;
    if (fprintf(params->fd,"\"nofInterKernel\": \"%u\",\n", params->nofKernels-1)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nofBlocks\": \"%u\",\n", kernelData[0].nofBlocks)  < 0 ) return -1;
    if (fprintf(params->fd,"\"nof_repetitions\": \"%d\",\n", params->nof_repetitions)  < 0 ) return -1;
    if (fprintf(params->fd,"\"data_size\": \"%d\",\n", params->data_size)  < 0 ) return -1;

    // Write times
    int size_time = 2 * kernelData[0].nofBlocks * params->nof_repetitions;

    if (fprintf(params->fd,"\"blocktimes\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n",params->blockTimes[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"],\n", params->blockTimes[size_time-1]) < 0 ) return -1;

    size_time = size_time / 2;
    if (fprintf(params->fd,"\"smid\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%u\",\n",params->smid[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%u\"],\n", params->smid[size_time-1]) < 0 ) return -1;

    size_time = params->nof_repetitions;
    if (fprintf(params->fd,"\"kernelDurations\":[\n") < 0 ) return -1;
    for (int i = 0; i < size_time-1; i++){
        if (fprintf(params->fd,"\"%lu\",\n",params->kernelDurations[i]) < 0 ) return -1;
    }
    if (fprintf(params->fd,"\"%lu\"]\n}", params->kernelDurations[size_time-1]) < 0 ) return -1;

    if (fclose(params->fd) < 0) return -1;
    return 0;
}

int cleanUp(param_t *params){
    kernel_data *kernelData = (kernel_data*)params->kernelData;

    for(int kernel = 0; kernel < params->nofKernels; kernel++){
        cudaStreamDestroy(kernelData[kernel].stream);
        if(kernelData[kernel].dummyData1 != NULL) cudaFree(kernelData[kernel].dummyData1);
        if(kernelData[kernel].dummyData2 != NULL) cudaFree(kernelData[kernel].dummyData2);
        if(kernelData[kernel].dummyData3 != NULL) cudaFree(kernelData[kernel].dummyData3);
        if(kernelData[kernel].targetTimes != NULL) cudaFree(kernelData[kernel].targetTimes);
        if(kernelData[kernel].smid != NULL) cudaFree(kernelData[kernel].smid);
    }

    // Free target buffers
    if(params->targetStartTime != NULL) cudaFree(params->targetStartTime);

    // Free host buffers
    if(params->blockTimes != NULL) free(params->blockTimes);
    if(params->smid != NULL) free(params->smid);
    if(params->kernelDurations != NULL) free(params->kernelDurations);
    if(params->kernelData != NULL) free(params->kernelData);

    cudaDeviceReset();
    return 0;
}
