#!/usr/bin/env python3
"""Benchmark for quickmp matrix profile computation with multiple devices and streams."""

import argparse
import sys
import threading
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
        "-d", "--devices", type=int, default=None,
        help="Number of devices to use (default: all available)"
    )
    parser.add_argument(
        "-s", "--streams", type=int, default=None,
        help="Number of streams per device (default: all available)"
    )
    args = parser.parse_args()

    print(f"Generating {args.count} time series of length {args.length}...")
    np.random.seed(42)
    timeseries_list = [np.random.rand(args.length) for _ in range(args.count)]

    quickmp.initialize()

    max_devices = quickmp.get_device_count()
    if args.devices is not None and args.devices > max_devices:
        quickmp.finalize()
        sys.exit(f"Error: Requested {args.devices} devices, but only {max_devices} available")
    num_devices = args.devices if args.devices else max_devices

    num_streams = None
    for d in range(num_devices):
        quickmp.use_device(d)
        max_streams = quickmp.get_stream_count()
        if args.streams is not None and args.streams > max_streams:
            quickmp.finalize()
            sys.exit(f"Error: Device {d} has only {max_streams} streams, but {args.streams} requested")
        if num_streams is None:
            num_streams = args.streams if args.streams else max_streams

    total_workers = num_devices * num_streams

    # Warm up all devices and streams
    for d in range(num_devices):
        quickmp.use_device(d)
        for s in range(num_streams):
            quickmp.selfjoin(timeseries_list[0], args.window, stream=s)

    # Create barrier for synchronization
    barrier = threading.Barrier(total_workers + 1)
    first_task_done = [False] * total_workers  # Track first task per worker

    def compute_mp(task):
        idx, T = task
        device_id = idx % num_devices
        stream_id = (idx // num_devices) % num_streams
        worker_id = device_id + stream_id * num_devices
        quickmp.use_device(device_id)

        # First task of each worker waits on barrier
        if not first_task_done[worker_id]:
            first_task_done[worker_id] = True
            barrier.wait()

        return quickmp.selfjoin(T, args.window, stream=stream_id)

    print(f"Computing matrix profiles with {num_devices} device(s) x {num_streams} stream(s) = {total_workers} workers...")

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(compute_mp, (i, T)) for i, T in enumerate(timeseries_list)]

        # Wait for all workers to be ready, then start timing
        barrier.wait()
        start = time.perf_counter()

        # Collect results
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start

    quickmp.finalize()

    print(f"Completed {args.count} matrix profiles in {elapsed:.3f} seconds")
    print(f"Throughput: {args.count / elapsed:.2f} profiles/sec")
    print(f"Average time per profile: {elapsed / args.count * 1000:.3f} ms")


if __name__ == "__main__":
    main()
