# improved-giggle
Some matrix multiplication implementations and benchmarking for CPUs and GPUs

## Usage

Make sure the `IMPROVED_GIGGLE_ROOT` environment variable is set to the root of the project


```
mkdir build && cd build
cmake ..
./apps/benchmark --benchmark {mlk,cublas,..} --number 1 --sizes 512,1024
```


I used llvm for openMP support on macOS:

```
cmake -DCMAKE_C_COMPILER="/usr/local/opt/llvm/bin/clang" -DCMAKE_CXX_COMPILER="/usr/local/opt/llvm/bin/clang++" ..
```

### Cluster Setup

CPU

```
module load cmake/3.12.1-fasrc01
module load gflags/2.1.2-fasrc01
module load intel/19.0.5-fasrc01
module load intel-mkl/2019.5.281-fasrc01
```

GPU

```
module load cmake/3.12.1-fasrc01
module load gflags/2.1.2-fasrc01
module load intel/19.0.5-fasrc01
module load cuda/10.1.243-fasrc01
```

## CPU implementations

cmake will attempt to find an MKL installation (`mkl`) and, if being compiled on macOS, will compile with the accelerate framework (`accelerate`) as a benchmark.

### Optimization One: `opt1CPU`

Implements the high level matrix decompositions described in [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf). 

### Optimization Two: `opt2CPU`

Relatively naive vectorization of previous version using explicit SIMD intrinsic functions.


### Optimization Three: `opt3CPU`

Further optimizes the inner kernel - adds a register blocking scheme to increase ratio of fp ops to loads. Still uses broadcasting in the vectorization.


### Optimization Four: `opt4CPU`

Stole some ideas from the Accelerate Framework `gemm` implementation after looking at its assembly. This replaced the broadcasting scheme with a permutation scheme. Ideally, the permutation scheme uses all 16 SIMD registers without spilling into the stack to minimize the number of loads/stores. However, I found that the compiler performed optimizations which increased the number of registers required beyond 16 and therefore decreased performance. The previous optimization uses 11 registers; this left room for the compiler use more registers without spilling into the stack.


### Optimization Five: `opt5CPU`

Being unable to convince the compiler to only use the 16 registers, I used inline assembly to implement the innermost kernel. I started with the assembly the compiler generated on `O1` (I think) and with loop unrolling disabled for the inner most loop. I changed the code to not use the extra registers and did some manual loop unrolling. This assembly is not optimally written with respect to the indexing and code alignment, but it seems to perform well.

### Optimization Six: `opt1OmpCPU`

The matrix multiplication was parallelized with openMP. To avoid race conditions, the parallelization happened at the highest level of the matrix decomposition, breaking the multiplication into a number of non-overlapping panel-panel multiplications.

## GPU Implementations

cmake will only build the GPU implementations if a CUDA compiler is detected. If this is the case, it will also enable `cublas` as an option.

### Optimization One: `opt1GPU`

The first two versions are from the CUDA developer documentation section on shared memory - with the naive version not using shared memory and the first optimization loading blocks of the matrix into shared memory. 


### Optimization Two: `opt2GPU`

The second optimization changes the blocking scheme to be close to the scheme in [Benchmarking GPUs to Tune Dense Linear Algebra](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf). This increases the register utilization and reduces the shared memory footprint. 


### Optimization Three: `opt3GPU`

The third optimization mostly builds off `opt1GPU` - adding a register blocking scheme that maintains coalesced memory access. 