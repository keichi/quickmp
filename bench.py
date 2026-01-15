#!/usr/bin/env python3
"""Benchmark for quickmp matrix profile computation with multiple streams."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import quickmp


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark quickmp matrix profile computation"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=1000,
        help="Number of time series to process (default: 1000)"
    )
    parser.add_argument(
        "-n", "--length", type=int, default=7200,
        help="Length of each time series (default: 7200)"
    )
    parser.add_argument(
        "-m", "--window", type=int, default=10,
        help="Subsequence window size (default: 10)"
    )
    parser.add_argument(
        "-s", "--streams", type=int, default=16,
        help="Number of streams for parallel execution (default: 16)"
    )
    args = parser.parse_args()

    print(f"Generating {args.count} time series of length {args.length}...")
    np.random.seed(42)
    timeseries_list = [np.random.rand(args.length) for _ in range(args.count)]

    quickmp.initialize()

    def compute_mp(task):
        idx, T = task
        stream_id = idx % args.streams
        return quickmp.selfjoin(T, args.window, stream=stream_id)

    print(f"Computing matrix profiles with {args.streams} streams...")
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.streams) as executor:
        results = list(executor.map(compute_mp, enumerate(timeseries_list)))

    elapsed = time.perf_counter() - start

    quickmp.finalize()

    print(f"Completed {args.count} matrix profiles in {elapsed:.3f} seconds")
    print(f"Throughput: {args.count / elapsed:.2f} profiles/sec")
    print(f"Average time per profile: {elapsed / args.count * 1000:.3f} ms")


if __name__ == "__main__":
    main()
