# complex-accelerate

GPU acceleration for complex-valued neural network training.

## Repository layout

- `experiments/`: small, reproducible comparisons of kernel implementations.
- `kernels/`: operation implementations, organized by operation and backend as needed.
- `correctness/`: reference computations and checks for kernel outputs.
- `results/` and `profiles/`: generated measurements and profiler artifacts (gitignored).

## First experiment

Test whether storing complex values as separate contiguous real and imaginary
tensors improves memory-access performance compared with interleaved storage.
This is a hypothesis to measure, not an assumed speedup.

Start with copy, elementwise addition, and elementwise complex multiplication on
vectors and matrices. Matrix multiplication is a separate operation for later.
Compare equivalent computations with the same shapes and component precision.
Keep layout conversion separate from the operation being measured.

Training and inference infrastructure will follow the initial acceleration work.

## Run the first kernels and profiler

Requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, a compatible local
CUDA toolkit (`nvcc`), a C++ compiler, and `ninja`. Install PyTorch for the target
machine using its official installation instructions; install Ninja with
`python -m pip install ninja`. The CUDA extension compiles on its first use into
PyTorch's extension cache. Compilation is outside timing.

From the repository root:

```bash
python -m unittest correctness.test_ops -v
python -m experiments.profile_ops
python -m experiments.profile_ops --shapes 257 1048576 1024x1024 --trace
```

The implementation supports contiguous complex64 interleaved tensors and pairs
of contiguous float32 tensors, with separate outputs and no autograd. Matrices
are traversed as flat contiguous storage. CUDA uses 256 threads per block and
disables fused multiply-add for this initial arithmetic comparison.

Each operation compares CUDA interleaved, CUDA split, eager PyTorch complex64,
and eager PyTorch split arithmetic. Outputs and multiplication scratch space
are preallocated. PyTorch split multiplication uses six operations, whereas
CUDA fuses it into one kernel; its speedup therefore includes fusion effects.
Use the two CUDA layouts to investigate storage layout itself. Copy and add
also use two PyTorch calls for split storage versus one custom CUDA launch.

The profiler checks outputs against complex128 arithmetic before timing, warms
up each case, randomizes case order across repeats, and saves individual CUDA
event batch-average times, host wall times, median latency, effective bandwidth,
and PyTorch speedup ratios to `results/`. Inputs are reused, so small cases may
be cache-resident. CUDA event intervals may include GPU idle time waiting for
Python/C++ submission; these are eager invocation timings, not pure kernel
durations. No performance threshold is assumed.

Effective bandwidth uses minimum logical traffic (16 bytes per complex element
for copy; 24 for add/multiply). It is not measured DRAM traffic, and does not
count PyTorch split intermediate traffic. `--trace` adds a separate CPU/CUDA
timeline and allocator-event trace under `profiles/`; it does not collect
hardware memory transaction counters. Detailed hardware analysis is deferred.

CPU baseline checks run without CUDA; GPU tests are explicitly skipped when
CUDA is unavailable. A passing CPU check does not validate CUDA compilation or
execution.
