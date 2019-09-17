#ifndef TESTBENCH_H
#define TESTBENCH_H
#include <cuda.h>
#include <cuda_runtime.h>

typedef enum {
            KER_SQR_NORM,
            KER_CONJ,
            KER_MULT,
            KER_GAUSS,
#if 0
            KER_SQR_MAG,
			KER_SUM_CHANN,
            KER_DIV,
            KER_ADD,
            KER_MUL_C,
            KER_ADD_C,
            KER_ELEM_MUL,
#endif
			KER_NO
} kernel_type_t;

inline const char* getKernelString(kernel_type_t type){
    switch(type){
    case KER_SQR_NORM:
        return "KER_SQR_NORM";
    case KER_CONJ:
        return "KER_CONJ";
    case KER_MULT:
        return "KER_MULT";
    case KER_GAUSS:
        return "KER_GAUSS";
    default:
        return "";
    }
}

typedef struct {
    int nofKernels;
    int32_t nof_repetitions;
    int data_size;
    uint64_t *blockTimes;
    uint64_t *kernelDurations;
    unsigned int *smid;
    uint64_t *targetStartTime;
    void * kernelData;
    FILE *fd;
    kernel_type_t kernelUT;
    kernel_type_t kernelInter;
} param_t;

int initializeTest(param_t *params);

int runTest(param_t *params);

int writeResults(param_t *params);

int cleanUp(param_t *params);
#endif
