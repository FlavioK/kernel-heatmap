#include <err.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "testbench.h"

static int startBenchmark(param_t *params) {
	cpu_set_t set;
	/* Ensure that our test thread does not migrate to another CPU
	 * during memguarding */
	CPU_ZERO(&set);
	CPU_SET(5, &set);

    if (initializeTest(params) < 0) return -1;
	

    // Run test
    if (runTest(params) < 0) return -1;

    // Write results
    if (writeResults(params) < 0){
        perror("Error while writing outpufile: ");
        return -1;
    }

    // Clean up
    if (cleanUp(params) < 0) return -1;

	return 0;
}

static void PrintUsage(const char *name) {
    printf("Usage: %s <measured kernel> <interfering kernel> <nof interfering kernels> <# of repetition> "
            "<output JSON file name>\n"
            "Available kernels: sqr_norm, conj, mult, gauss\n", name);
}

static kernel_type_t getType(char* kernel){

	if (strcmp(kernel, "sqr_norm") == 0) {
	  return KER_SQR_NORM;

    } else if (strcmp(kernel, "conj") == 0) {
      return KER_CONJ;

    } else if (strcmp(kernel, "mult") == 0) {
      return KER_MULT;

    } else if (strcmp(kernel, "gauss") == 0){
      return KER_GAUSS;
#if 0
    } else if (strcmp(kernel, "sqr_mag") == 0) {
	  return KER_SQR_MAG;


	} else if (strcmp(kernel, "sum_ch") == 0) {
	  return KER_SUM_CHANN;


	} else if (strcmp(kernel, "div") == 0) {
	  return KER_DIV;

	} else if (strcmp(kernel, "add") == 0) {
	  return KER_ADD;

	} else if (strcmp(kernel, "mul_c") == 0) {
	  return KER_MUL_C;

	} else if (strcmp(kernel, "add_c") == 0) {
	  return KER_ADD_C;

	} else if (strcmp(kernel, "elemmul") == 0) {
	  return KER_ELEM_MUL;

#endif
    } else {
		return KER_NO;
	}
}

int main(int argc, char **argv) {

    if (argc != 6) {
        PrintUsage(argv[0]);
        return 1;
    }

    param_t params;

    // Parse input parameter
	kernel_type_t type = getType(argv[1]);
	if (type == KER_NO ){
        printf("Kernel not available. Got %s\n", argv[1]);
        return EXIT_FAILURE;
	}
    params.kernelUT = type;

	type = getType(argv[2]);
	if (type == KER_NO ){
        printf("Kernel not available. Got %s\n", argv[2]);
        return EXIT_FAILURE;
	}
    params.kernelInter = type;

    int nof_kernel = atoi(argv[3]);
    if (nof_kernel < 0) {
        printf("Min 1 kernel. Got %s kernels\n", argv[3]);
        return EXIT_FAILURE;
    }

    params.nofKernels = nof_kernel + 1;
    params.data_size = 366080;

    int nof_repetitions = atoi(argv[4]);
        if (nof_repetitions <= 0) {
        printf("More than 0 repetitions need to be used. Got %s repetitions\n", argv[4]);
        return EXIT_FAILURE;
    }
    params.nof_repetitions = nof_repetitions;
    
    params.fd = NULL;
    params.fd = fopen(argv[5],"w");
    if (params.fd == NULL) {
        perror("Error opening output file:");
        return EXIT_FAILURE;
    }

    startBenchmark(&params);
    printf("Finished testrun\n");
    return 0;
}
