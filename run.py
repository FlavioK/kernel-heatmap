#!/usr/bin/python3
from subprocess import Popen, PIPE
import json
import statistics as st
import sys
import pathlib
import shutil

outpath = "out/"
pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
shutil.chown(outpath, "nvidia", "nvidia");


#Usage: ./testbench <kernel1> <kernel2> <nof kernel2> <# of repetition> <usePREM 1/0><output JSON file name>

kernels = ['sqr_norm', 'conj', 'mult', 'gauss']
nof_rep        = 1000

for kernel1 in kernels:
    for kernel2 in kernels:
        for nof_kernel2 in range(3): # Max 2 interfering kernels
        
            print("-------------------------------")
            print("Measured kernel: "+kernel1)
            print("Interfering kernels: "+kernel2)
            print("Number of kernel2: "+str(nof_kernel2))
            print("-------------------------------")

            process = Popen(["chrt" ,"-f", "99", "build/testbench", kernel1, kernel2, str(nof_kernel2), str(nof_rep) , "out.json"], stdout=PIPE)
            output = process.communicate()
            print(output)

            # Copy the output file
            newFile =  outpath+"{:s}-{:s}-numberinterfering-{:d}-1000rep.json".format(kernel1, kernel2, nof_kernel2)
            shutil.move('out.json', newFile) # complete target filename given
            shutil.chown(newFile, "nvidia", "nvidia");
