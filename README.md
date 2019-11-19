# improved-giggle
Linear algebra benchmarking and implementations

## Usage

Make sure the `IMPROVED_GIGGLE_ROOT` environment variable is set to the root of the project

`set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")`

`sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'`

`toplev.py -d -l3 ../improved-giggle/build/apps/benchmark --sizes 1024 --benchmark opt1CPU`

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


### Opt1

```
RET            Retiring             % Slots                  96.67
RET            Retiring.Base        % Slots                  96.63
RET            Retiring.Base.Other  % Uops                   95.81  <==
	This metric represents non-floating-point (FP) uop fraction
	the CPU has executed...
MUX                                 %                         5.25
	PerfMon Event Multiplexing accuracy indicator
```

### Opt2

```
BE             Backend_Bound                               % Slots                  31.96
BE/Core        Backend_Bound.Core_Bound                    % Slots                  27.38
BE/Core        Backend_Bound.Core_Bound.Ports_Utilization  % Clocks                 28.12  <==
	This metric estimates fraction of cycles the CPU performance
	was potentially limited due to Core computation issues (non
	divider-related)...
MUX                                                        %                         5.19
	PerfMon Event Multiplexing accuracy indicator
warning: 6 results not referenced: 62 65 66 82 83 84
```
