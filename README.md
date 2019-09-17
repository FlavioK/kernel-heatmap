# KCF-Tracker GaussianCorrelation kernel heat-map

This project is used to create a heatmap of the GaussianCorrealtion kernels
used in the [kcf-tracker](https://github.com/CTU-IIG/kcf). It runs a specified
multiple times with a requested number of interfering kernels. The kernel
durations, recorded with the CUDA-globaltimer register, are logged to a file
and postprocessed by a python3 script to create a heatmap.

## Dependencies
```
sudo apt install cmake ninja-build
```

Jetson TX2 with CUDA installed and the original nvidia user with password nvidia.

## Build the project on the Jetson TX2

To build the project on the target just clone this repository to the nvidia
home directory and call:

```
make all
```

This builds the executable `testbench` in the directory `build`. To run the
whole benchmark use the command:

```
export LD_BIND_NOW && echo 'nvidia' | sudo -S python3 run.py
```
This command runs multiple times the executable `testbench` with different
kernel configurations and stores the resulting files in the directory `out`.

To visualize the heatmap call:
```
python3 dataproc/heatMap.py
```

## Call the benchmark from local host and run it on the target

Clone the repository to your local machine and use the following command:

```
make target_run
```

This command deploys the necessary files to the Jetson TX2, builds the project,
runs the benchmarks and copies back the files to the host.

To visualize the heatmap call:
```
python3 dataproc/heatMap.py
```
