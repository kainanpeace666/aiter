import math

import torch
import triton

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

def _estimate_pool_size(M, N, K, dtype, target_mb=256):
    per_set_bytes = (M * K + N * K + M * N) * dtype.itemsize
    return max(4, math.ceil(target_mb * 1024 * 1024 / per_set_bytes))

def _create_buffer_pool(M, N, K, pool_size, layout, use_bias, dtype=torch.bfloat16):
    pool = []
    for _ in range(pool_size):
        x, w, bias, _out_dtype, y = generate_gemm_a16w16_inputs(
            M, N, K, dtype, layout=layout, output=True, bias=use_bias
        )
        pool.append((x, w, bias, y))
    return pool

def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str,
    atomic: bool = False,
    activation: str | None = None,
    persistent: bool = False,
    kernel_type: str = "bandwidth_bound",
    **kwargs,
):
    # NOTE: Assume bias and output has the same dtype
    c_dtype = torch.bfloat16
    x, w, bias, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, c_dtype, layout=layout, output=True, bias=True
    )
    # flops
    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N  # elementwise ops on the GEMM output
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
    mem_write = (M * N) * x.element_size()
    mem = mem_read + mem_write

    bench_fn = triton.testing.do_bench_cudagraph

    if atomic:
        # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
        assert backend != "gluon", "Atomic kernel is triton-only"
        assert (
            activation is None
        ), "Atomic kernel does not currently support fused activation"
        y = y.to(torch.float32).zero_()
        ms = bench_fn(
            lambda: gemm_a16w16_atomic(x, w, torch.float32, y),
        )
    elif persistent:
        assert not (backend == "gluon" and layout not in ("TN", "TT")), (
            f"--persistent with --backend gluon requires --layout TN or TT, got "
            f"'{layout}' (the gluon persistent kernel has no 'N' path for A)"
        )
        ms = bench_fn(
            lambda: gemm_a16w16(
                x,
                w,
                bias,
                c_dtype,
                y,
                activation=activation,
                kernel_type=kernel_type,
                backend=backend,
                persistent=True,
            ),
        )
    else:
        ms = bench_fn(
            lambda: gemm_a16w16(
                x,
                w,
                bias,
                c_dtype,
                y,
                activation=activation,
                kernel_type=kernel_type,
                backend=backend,
            ),
        )

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)

def bench_gemm_fn_rotating(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    backend: str,
    atomic: bool = False,
    activation: str | None = None,
    persistent: bool = False,
    kernel_type: str = "bandwidth_bound",
    use_bias: bool = False,
    kernels_per_graph: int = 1000,
    num_replays: int = 10,
    warmup_replays: int = 3,
    pool_size: int | None = None,
    target_pool_mb: int = 256,
    **kwargs,
):
    c_dtype = torch.bfloat16

    flops = 2.0 * M * N * K
    if activation is not None:
        flops += M * N
    mem_read = (M * K) * c_dtype.itemsize + (N * K) * c_dtype.itemsize
    mem_write = (M * N) * c_dtype.itemsize
    mem = mem_read + mem_write


    if pool_size is None:
        pool_size = _estimate_pool_size(
            M, N, K, dtype=c_dtype, target_mb=target_pool_mb
        )

    pool = _create_buffer_pool(M, N, K, pool_size, layout, use_bias, dtype=c_dtype)
    per_set_mb = ((M * K + N * K + M * N) * c_dtype.itemsize) / (1024*1024)
    total_pool_mb = per_set_mb * pool_size
    print(
        f"  Rotating buffer pool: {pool_size} sets x {per_set_mb:.2f} MB "
        f"= {total_pool_mb:.1f} MB total"
    )
    print(
        f"  Kernels/graph: {kernels_per_graph}, "
        f"replays: {num_replays} (+{warmup_replays} warmup)"
    )

    # Warmup: trigger JIT compilation outside of graph capture
    x, w, bias, y = pool[0]
    for _ in range(3):
        gemm_a16w16(
            x, w, bias, c_dtype, y,
            activation=activation,
            kernel_type=kernel_type,
            backend=backend,
            persistent=persistent,
        )
    torch.cuda.synchronize()

    # Build per-dispatch callables that rotate through the buffer pool
    def _make_fn(idx):
        x, w, bias, y = pool[idx % pool_size]
        return lambda: gemm_a16w16(
            x, w, bias, c_dtype, y,
            activation=activation,
            kernel_type=kernel_type,
            backend=backend,
            persistent=persistent,
        )

    fns = [_make_fn(i) for i in range(kernels_per_graph)]

    #Capture graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for fn in fns[:min(3, len(fns))]:
            fn()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for fn in fns:
                fn()
    torch.cuda.synchronize()

    #Warmup replays
    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()

    #Timed replays
    start_events = []
    end_events = []
    for _ in range(num_replays):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        g.replay()
        ee.record()
        start_events.append(se)
        end_events.append(ee)
    torch.cuda.synchronize()

    times_ms = [se.elapsed_time(ee) for se, ee in zip(start_events, end_events)]
    per_kernel_ms = [t / kernels_per_graph for t in times_ms]

    ms = sum(per_kernel_ms) / len(per_kernel_ms)

    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)

def run_model_benchmark(args, backend):
    benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, hidden_dim, intermediate_dim, metric, layer, **kwargs):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N, args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K, args.tp)
        if args.r:
            return bench_gemm_fn_rotating(
                M, N, K, metric,
                args.layout, backend,
                activation=args.activation,
                persistent=args.persistent,
                kernel_type=args.kernel_type,
                use_bias=args.bias,
                kernels_per_graph=args.kernels_per_graph,
                num_replays=args.num_replays,
                warmup_replays=args.warmup_replays,
                pool_size=args.pool_size,
                target_pool_mb=args.target_pool_mb,
            )
        else:
            return bench_gemm_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            backend,
            atomic=args.atomic,
            activation=args.activation,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
            )
    
    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, backend):
    """
    Runs a benchmark with given tensor shapes.
    """
    benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a16w16(M, N, K, metric, **kwargs):
        N = math.ceil(N / args.tp)
        if args.r:
            return bench_gemm_fn_rotating(
                M, N, K, metric,
                args.layout, backend,
                activation=args.activation,
                persistent=args.persistent,
                kernel_type=args.kernel_type,
                use_bias=args.bias,
                kernels_per_graph=args.kernels_per_graph,
                num_replays=args.num_replays,
                warmup_replays=args.warmup_replays,
                pool_size=args.pool_size,
                target_pool_mb=args.target_pool_mb,
            )
        else:
            return bench_gemm_fn(
            M,
            N,
            K,
            metric,
            args.layout,
            backend,
            atomic=args.atomic,
            activation=args.activation,
            persistent=args.persistent,
            kernel_type=args.kernel_type,
        )

    bench_gemm_a16w16.run(save_path="." if args.o else None, print_data=True)

def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"
    
    if args.backend == "gluon":
        assert _is_gluon_available(), "Gluon backend not available"
    
    print(f"Using backend: {args.backend}")
    backend = args.backend if args.backend is not "auto" else "None"

    if args.model:
        run_model_benchmark(args, backend)
    else:
        unsupported_args = [
            "fc1",
            "fc2",
            "no_glu",
        ]
        for arg in unsupported_args:
            if getattr(args, arg, None) != getattr(defaults, arg, None):
                raise RuntimeError(
                    f"Argument '{arg}' is not supported for benchmarking without the --model flag."
                )
        run_shape_benchmark(args, backend)

def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A16W16 GEMM")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="Use the atomic kernel (split-k with atomic_add) instead of the standard a16w16 kernel.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function to apply to the output. One of ('gelu', 'gelu_tanh', 'silu', 'silu_exp2', 'relu').",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        default=False,
        help="Use the persistent kernel (gemm_a16w16(..., persistent=True)) instead of "
        "the standard a16w16 kernel",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["bandwidth_bound", "compute_bound"],
        default="bandwidth_bound",
        help="Kernel variant to use (gluon only). Default: bandwidth_bound.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Enable bias in the GEMM (ADD_BIAS=1). Default: no bias (ADD_BIAS=0).",
    )
    parser.add_argument(
        "--kernels-per-graph",
        type=int,
        default=1000,
        help="Number of kernel dispatches captured per graph.",
    )
    parser.add_argument(
        "--num-replays",
        type=int,
        default=10,
        help="Number of timed graph replays.",
    )
    parser.add_argument(
        "--warmup-replays",
        type=int,
        default=3,
        help="Number of untimed warmup graph replays.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Override buffer pool size. Auto-sized to exceed L2 by default.",
    )
    parser.add_argument(
        "--target-pool-mb",
        type=int,
        default=256,
        help="Target total pool size in MB when auto-sizing.",
    )
    return get_ff_args(parser, args=args)

def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)

if __name__ == "__main__":
    main()