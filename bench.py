#!/usr/bin/env python3
"""Benchmark for quickmp batched matrix profile computation with multiple devices."""

import argparse
import sys
import threading
import time

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
        "-b", "--batch", type=int, default=64,
        help="Number of time series per selfjoin_batch call (default: 64)"
    )
    args = parser.parse_args()

    print(f"Generating {args.count} time series of length {args.length}...")
    np.random.seed(42)
    T = np.random.rand(args.count, args.length)

    quickmp.initialize()

    max_devices = quickmp.get_device_count()
    if args.devices is not None and args.devices > max_devices:
        quickmp.finalize()
        sys.exit(f"Error: Requested {args.devices} devices, but only {max_devices} available")
    num_devices = args.devices if args.devices else max_devices

    # Warm up all devices
    for d in range(num_devices):
        quickmp.use_device(d)
        quickmp.selfjoin_batch(T[:args.batch], args.window)

    # Each device processes a contiguous share of the time series in batches
    shares = np.array_split(np.arange(args.count), num_devices)
    barrier = threading.Barrier(num_devices + 1)

    def worker(device_id, indices):
        quickmp.use_device(device_id)
        barrier.wait()
        for start in range(0, len(indices), args.batch):
            idx = indices[start:start + args.batch]
            quickmp.selfjoin_batch(T[idx[0]:idx[-1] + 1], args.window)

    print(f"Computing matrix profiles with {num_devices} device(s), batch size {args.batch}...")

    threads = [threading.Thread(target=worker, args=(d, shares[d])) for d in range(num_devices)]
    for t in threads:
        t.start()

    barrier.wait()
    start = time.perf_counter()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start

    quickmp.finalize()

    print(f"Completed {args.count} matrix profiles in {elapsed:.3f} seconds")
    print(f"Throughput: {args.count / elapsed:.2f} profiles/sec")
    print(f"Average time per profile: {elapsed / args.count * 1000:.3f} ms")


if __name__ == "__main__":
    main()
