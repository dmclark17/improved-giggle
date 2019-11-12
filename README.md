# improved-giggle
Linear algebra benchmarking and implementations

## Usage

Make sure the `IMPROVED_GIGGLE_ROOT` environment variable is set to the root of the project

```
./apps/benchmark --benchmark {mlk,cublas} --number 1 --sizes 512,1024
```


## CPU implementations

### Naive

```
BE             Backend_Bound                          % Slots                  89.48
BE/Mem         Backend_Bound.Memory_Bound             % Slots                  77.10
BE/Core        Backend_Bound.Core_Bound               % Slots                  12.39
	This metric represents fraction of slots where Core non-
	memory issues were of a bottleneck...
BE/Mem         Backend_Bound.Memory_Bound.DRAM_Bound  % Stalls                 72.58  <==
	This metric estimates how often the CPU was stalled on
	accesses to external memory (DRAM) by loads...
	Sampling events:  mem_load_retired.l3_miss:pp
MUX                                                   %                         5.26
	PerfMon Event Multiplexing accuracy indicator
```
