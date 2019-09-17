#ifndef UTIL_FUNC_H
#define UTIL_FUNC_H
#include <stdint.h>
#include <cuda.h>
#include <stdint.h>

#define START_TIME_OFFSET_NS (0)//(20000000) //10ms


static __device__ __inline__ uint64_t getTime(void){
    uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

static __device__ __inline__ unsigned int get_smid(void)
{
    unsigned int ret;
    asm("mov.u32 %0, %%smid;":"=r"(ret) );
    return ret;
}


static __device__ __inline__ void spinUntil(uint64_t endTime){
    if( threadIdx.x == 0){
        while(getTime() < endTime);
    }
}

static __device__ __inline__ void logBlockStart(const uint64_t startTime, uint64_t *targetTimes, unsigned int *smid){
   
    // Spin until PREM schedule start time 
    spinUntil(startTime);

    uint64_t start_time = getTime();
    if(threadIdx.x == 0){
        targetTimes[blockIdx.x*2] = start_time;
        smid[blockIdx.x] = get_smid();
}
}

static __device__ __inline__ void logBlockEnd(uint64_t *targetTimes){
    if(threadIdx.x == 0){
        targetTimes[blockIdx.x*2+1] = getTime();
    }
}

#endif
