#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// Each thread handles both components. Both layouts use one launch per operation.
template<int Op, bool Split>
__global__ void elementwise(const float* x, const float* xi,
                            const float* y, const float* yi,
                            float* out, float* oi, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += int64_t(blockDim.x) * gridDim.x) {
        float2 a;
        if constexpr (Split) a = make_float2(x[i], xi[i]);
        else a = reinterpret_cast<const float2*>(x)[i];
        float2 z = a;
        if constexpr (Op != 0) {
            float2 b;
            if constexpr (Split) b = make_float2(y[i], yi[i]);
            else b = reinterpret_cast<const float2*>(y)[i];
            if constexpr (Op == 1) z = make_float2(a.x + b.x, a.y + b.y);
            else z = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
        if constexpr (Split) { out[i] = z.x; oi[i] = z.y; }
        else reinterpret_cast<float2*>(out)[i] = z;
    }
}

void run(int op, bool split, std::vector<torch::Tensor> inputs,
         std::vector<torch::Tensor> outputs) {
    TORCH_CHECK(op >= 0 && op <= 2, "op must be 0 (copy), 1 (add), or 2 (multiply)");
    const size_t components = split ? 2 : 1;
    TORCH_CHECK(inputs.size() == components * (op == 0 ? 1 : 2), "wrong input count");
    TORCH_CHECK(outputs.size() == components, "wrong output count");
    const auto& first = inputs[0];
    const auto dtype = split ? torch::kFloat32 : torch::kComplexFloat;
    auto validate = [&](const torch::Tensor& t) {
        TORCH_CHECK(t.is_cuda() && t.device() == first.device(), "tensors must share a CUDA device");
        TORCH_CHECK(t.scalar_type() == dtype && t.is_contiguous(), "unsupported dtype or noncontiguous tensor");
        TORCH_CHECK(t.sizes() == first.sizes(), "shapes must match");
        TORCH_CHECK(!t.requires_grad(), "experimental kernels do not support autograd");
        TORCH_CHECK(!t.is_conj() && !t.is_neg(), "resolve conjugate/negative views first");
        TORCH_CHECK(split || reinterpret_cast<uintptr_t>(t.data_ptr()) % alignof(float2) == 0,
                    "interleaved storage must be float2 aligned");
    };
    for (const auto& t : inputs) validate(t);
    for (const auto& t : outputs) {
        validate(t);
        for (const auto& input : inputs) at::assert_no_overlap(t, input);
    }
    if (split) at::assert_no_overlap(outputs[0], outputs[1]);
    c10::cuda::CUDAGuard guard(first.device());
    int64_t n = first.numel();
    if (!n) return;
    auto ptr = [](const torch::Tensor& t) { return static_cast<float*>(t.data_ptr()); };
    const float* x = ptr(inputs[0]);
    const float* xi = split ? ptr(inputs[1]) : nullptr;
    const float* y = op ? ptr(inputs[components]) : nullptr;
    const float* yi = op && split ? ptr(inputs[3]) : nullptr;
    float* out = ptr(outputs[0]);
    float* oi = split ? ptr(outputs[1]) : nullptr;
    const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65535));
    auto stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH(OP, SPLIT) elementwise<OP, SPLIT><<<blocks, 256, 0, stream>>>(x, xi, y, yi, out, oi, n)
    if (split) {
        if (op == 0) { LAUNCH(0, true); }
        else if (op == 1) { LAUNCH(1, true); }
        else { LAUNCH(2, true); }
    } else {
        if (op == 0) { LAUNCH(0, false); }
        else if (op == 1) { LAUNCH(1, false); }
        else { LAUNCH(2, false); }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "FP32 component copy/add/multiply (out-of-place, no autograd)");
}
