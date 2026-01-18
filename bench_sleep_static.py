#!/usr/bin/env python3
"""Benchmark to measure VEDA/VEO overhead using sleep kernel (static distribution)."""

import argparse
import sys
import threading
import time

import quickmp


def main():
    parser = argparse.ArgumentParser(description="Sleep benchmark for quickmp (static)")
    parser.add_argument("-c", "--count", type=int, default=1000,
                        help="Number of sleep tasks (default: 1000)")
    parser.add_argument("-u", "--microseconds", type=int, default=1000,
                        help="Sleep duration in microseconds (default: 1000)")
    parser.add_argument("-d", "--devices", type=int, default=None,
                        help="Number of devices to use (default: all available)")
    parser.add_argument("-s", "--streams", type=int, default=None,
                        help="Number of streams per device (default: all available)")
    args = parser.parse_args()

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

    # Warm up
    for d in range(num_devices):
        quickmp.use_device(d)
        for s in range(num_streams):
            quickmp.sleep_us(100, s)

    def worker_func(worker_id, device_id, stream_id, task_count):
        """Worker thread function with fixed device and stream."""
        quickmp.use_device(device_id)
        for _ in range(task_count):
            quickmp.sleep_us(args.microseconds, stream_id)

    # Distribute tasks statically to workers
    base_tasks = args.count // total_workers
    remainder = args.count % total_workers
    worker_tasks = [base_tasks + (1 if i < remainder else 0) for i in range(total_workers)]

    # Run benchmark
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

    # Calculate expected time
    expected = args.count * args.microseconds / 1e6 / total_workers
    efficiency = expected / elapsed * 100

    print(f"Completed {args.count} sleep tasks ({args.microseconds}us each) "
          f"with {num_devices} device(s) x {num_streams} stream(s) = {total_workers} workers (static) in {elapsed:.3f} seconds")
    print(f"Expected time (ideal): {expected:.3f}s")
    print(f"Parallel efficiency: {efficiency:.1f}%")

    quickmp.finalize()


if __name__ == "__main__":
    main()
