#!/usr/bin/env python3
"""Benchmark with static task distribution (no ThreadPoolExecutor)."""

import argparse
import sys
import threading
import time

import numpy as np
import quickmp


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark quickmp with static task distribution"
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
    results = [None] * args.count

    def worker_func(worker_id, device_id, stream_id, task_indices):
        """Worker thread function with fixed device and stream."""
        quickmp.use_device(device_id)
        for idx in task_indices:
            results[idx] = quickmp.selfjoin(timeseries_list[idx], args.window, stream=stream_id)

    # Distribute tasks statically to workers
    # Worker i gets tasks i, i+total_workers, i+2*total_workers, ...
    worker_tasks = [[] for _ in range(total_workers)]
    for idx in range(args.count):
        worker_id = idx % total_workers
        worker_tasks[worker_id].append(idx)

    print(f"Computing matrix profiles with {num_devices} device(s) x {num_streams} stream(s) = {total_workers} workers (static)...")
    start = time.perf_counter()

    threads = []
    for worker_id in range(total_workers):
        device_id = worker_id % num_devices
        stream_id = worker_id // num_devices
        t = threading.Thread(
            target=worker_func,
            args=(worker_id, device_id, stream_id, worker_tasks[worker_id])
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start

    quickmp.finalize()

    print(f"Completed {args.count} matrix profiles in {elapsed:.3f} seconds")
    print(f"Throughput: {args.count / elapsed:.2f} profiles/sec")
    print(f"Average time per profile: {elapsed / args.count * 1000:.3f} ms")


if __name__ == "__main__":
    main()
