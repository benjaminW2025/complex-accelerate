"""Run from the repository root: python -m experiments.profile_ops --help."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time


def shape_arg(value):
    try:
        shape = tuple(int(x) for x in value.split("x"))
        if len(shape) not in (1, 2) or any(x < 1 for x in shape):
            raise ValueError
        return shape
    except ValueError:
        raise argparse.ArgumentTypeError("use positive N or rowsxcols") from None


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_cases(torch, extension, x, y, op):
    """Allocate all outputs, conversions, and scratch outside measured regions."""
    code = {"copy": 0, "add": 1, "multiply": 2}[op]
    xr, xi = x.real.contiguous(), x.imag.contiguous()
    yr, yi = y.real.contiguous(), y.imag.contiguous()
    cases = []
    for backend in ("cuda", "torch"):
        for layout in ("interleaved", "split"):
            split = layout == "split"
            outputs = [torch.empty_like(xr), torch.empty_like(xi)] if split else [torch.empty_like(x)]
            inputs = [xr, xi] if split else [x]
            if code:
                inputs += [yr, yi] if split else [y]
            if backend == "cuda":
                def run(inputs=inputs, outputs=outputs, split=split):
                    extension.run(code, split, inputs, outputs)
            elif not split:
                def run(out=outputs[0]):
                    if code == 0:
                        out.copy_(x)
                    elif code == 1:
                        torch.add(x, y, out=out)
                    else:
                        torch.mul(x, y, out=out)
            else:
                scratch = torch.empty_like(xr) if code == 2 else None

                def run(outputs=outputs, scratch=scratch):
                    real, imag = outputs
                    if code == 0:
                        real.copy_(xr)
                        imag.copy_(xi)
                    elif code == 1:
                        torch.add(xr, yr, out=real)
                        torch.add(xi, yi, out=imag)
                    else:
                        torch.mul(xr, yr, out=real)
                        torch.mul(xi, yi, out=scratch)
                        torch.sub(real, scratch, out=real)
                        torch.mul(xr, yi, out=imag)
                        torch.mul(xi, yr, out=scratch)
                        torch.add(imag, scratch, out=imag)
            cases.append((f"{backend}_{layout}", run, outputs))
    return cases


def check_case(torch, run, outputs, reference, op):
    run()
    actual = torch.complex(*outputs) if len(outputs) == 2 else outputs[0]
    expected = reference.to(torch.complex64)
    # Copy must preserve values exactly; arithmetic allows FP32 rounding.
    torch.testing.assert_close(actual, expected, rtol=0 if op == "copy" else 2e-5,
                               atol=0 if op == "copy" else 2e-6)
    return (actual.to(torch.complex128) - reference).abs().max().item() if actual.numel() else 0.0


def measure(torch, run, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations, (time.perf_counter() - wall_start) * 1e6 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=shape_arg, default=[(1024,), (1048576,), (1024, 1024)])
    parser.add_argument("--ops", nargs="+", choices=["copy", "add", "multiply"], default=["copy", "add", "multiply"])
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="collect CPU/CUDA trace in a separate pass")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        import torch
        from kernels.ops import load_ops
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use an NVIDIA GPU with CUDA-enabled PyTorch and the CUDA toolkit.")
        torch.cuda.set_device(args.device)
        extension = load_ops()
    except (ImportError, RuntimeError) as error:
        parser.exit(1, f"Setup failed: {error}\n")

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or root / "results" / f"ops-{stamp}.json"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    report = {"metadata": {
        "timestamp_utc": stamp, "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "git_revision": revision.stdout.strip(), "git_dirty": bool(status.stdout.strip()),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "timing": "CUDA event batch average; includes GPU idle gaps from host submission",
        "bandwidth": "algorithmic minimum bytes / event time; not measured DRAM traffic",
        "compiler_flags": ["-O3", "--fmad=false"],
    }, "results": []}

    with torch.inference_mode():
        for shape in args.shapes:
            x = torch.randn(shape, device="cuda", dtype=torch.complex64)
            y = torch.randn_like(x)
            for op in args.ops:
                xd, yd = x.to(torch.complex128), y.to(torch.complex128)
                reference = xd if op == "copy" else xd + yd if op == "add" else xd * yd
                cases = make_cases(torch, extension, x, y, op)
                errors, samples, walls = {}, {}, {}
                for name, run, outputs in cases:
                    errors[name] = check_case(torch, run, outputs, reference, op)
                    for _ in range(args.warmup):
                        run()
                    samples[name], walls[name] = [], []
                for _ in range(args.repeats):
                    order = cases.copy()
                    rng.shuffle(order)
                    for name, run, _ in order:
                        gpu_us, wall_us = measure(torch, run, args.iterations)
                        samples[name].append(gpu_us)
                        walls[name].append(wall_us)
                for name, run, _ in cases:
                    median = statistics.median(samples[name])
                    layout = name.split("_", 1)[1]
                    row = {"op": op, "shape": shape, "case": name,
                           "event_us_samples": samples[name], "event_us_median": median,
                           "wall_us_samples": walls[name], "correctness_max_abs_error": errors[name],
                           "effective_GB_s": math.prod(shape) * (16 if op == "copy" else 24) / median / 1000,
                           "speedup_vs_torch_same_layout": statistics.median(samples[f"torch_{layout}"]) / median,
                           "speedup_vs_torch_interleaved": statistics.median(samples["torch_interleaved"]) / median}
                    if args.trace:
                        trace = root / "profiles" / stamp / f"{op}-{'x'.join(map(str, shape))}-{name}.json"
                        trace.parent.mkdir(parents=True, exist_ok=True)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True, profile_memory=True) as prof:
                            with torch.profiler.record_function(name):
                                run()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(trace))
                        row["trace"] = str(trace)
                    report["results"].append(row)
                    print(f"{op:8} {str(shape):16} {name:18} {median:10.3f} us  {row['effective_GB_s']:9.2f} effective GB/s")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
