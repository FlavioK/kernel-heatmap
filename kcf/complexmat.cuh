#ifndef COMPLEX_MAT_HPP_213123048309482094
#define COMPLEX_MAT_HPP_213123048309482094

#include <stdint.h>
#include <cuda.h>

struct kernel_data{
    unsigned int nofBlocks;
    unsigned int nofThreads;
    float *dummyData1;
    float *dummyData2;
    float *dummyData3;
    unsigned int dataSize;
    void (*kernelFunc)(kernel_data *);
    int *interfering;
    uint64_t *targetTimes;
    uint64_t *startTime;
    unsigned int *smid;
    cudaStream_t stream;
};

typedef void (*kernelFunc)(kernel_data *);

void sqr_norm(kernel_data *data);
void conj(kernel_data *data);
void same_num_channels_mul(kernel_data *data);
void GaussianCorrelation(kernel_data *data);



#endif // COMPLEX_MAT_HPP_213123048309482094
