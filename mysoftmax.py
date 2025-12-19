import time
from functools import partial
from typing import Optional
import os
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="softmax_lib",
    sources=["mysoftmax.cu"],
    # build_directory="./build",
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    # verbose=True,
    extra_cflags=["-std=c++17"],
)

def run_golden_softmax(a: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        out = torch.zeros_like(a)
    out = torch.softmax(a, dim=-1)
    return out

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # torch.dot vs custom dot_prod kernel
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, out)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time

# Ss = [1024, 2048, 4096]
# Ks = [1024, 2048, 4096]
# SKs = [(S, K) for S in Ss for K in Ks]

SKs = [(3, 4)]

for item in SKs:
    S = item[0]
    K = item[1]
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    a = torch.randn((S, K)).cuda().float().contiguous()
    c = torch.zeros((S, K)).cuda().float().contiguous()
    run_benchmark(lib.softmax_baseline, a,  "f32", c)
    output = run_golden_softmax(a)
    print("Result Max Diff: ", torch.max(abs(output-c)))
    

    # print("-" * 85)
    # a_f16 = a.half().contiguous()
    # b_f16 = b.half().contiguous()
    # c_f16 = c.half().contiguous()
    # run_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", c_f16)
    # run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", c_f16)
    # run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
    # run_benchmark(
    #     lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16
    # )
    # run_benchmark(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_th")
    print("-" * 85)