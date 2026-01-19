#!/usr/bin/env python3
"""Sleep benchmark with one process per device to avoid VEDA lock contention."""

import argparse
import multiprocessing as mp
import sys
import time


def worker_process(device_id, num_streams, task_count, microseconds, result_queue, barrier):
    """Worker process that handles one VE device."""
    import threading
    import quickmp

    # Initialize only this device
    quickmp.initialize(device_start=device_id, device_count=1)
    quickmp.use_device(0)  # Device 0 is the only initialized device

    # Warm up
    for s in range(num_streams):
        quickmp.sleep_us(100, s)

    # Distribute tasks to streams
    base_tasks = task_count // num_streams
    remainder = task_count % num_streams
    stream_task_counts = [base_tasks + (1 if i < remainder else 0) for i in range(num_streams)]

    def stream_worker(stream_id, count):
        quickmp.use_device(0)  # Each thread needs to set device (thread_local)
        for _ in range(count):
            quickmp.sleep_us(microseconds, stream_id)

    # Wait for all processes to be ready
    barrier.wait()

    # Execute with threads (one per stream)
    threads = []
    for stream_id in range(num_streams):
        t = threading.Thread(target=stream_worker, args=(stream_id, stream_task_counts[stream_id]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    quickmp.finalize()
    result_queue.put(task_count)


def main():
    parser = argparse.ArgumentParser(
        description="Sleep benchmark with one process per device"
    )
    parser.add_argument("-c", "--count", type=int, default=1000,
                        help="Number of sleep tasks (default: 1000)")
    parser.add_argument("-u", "--microseconds", type=int, default=1000,
                        help="Sleep duration in microseconds (default: 1000)")
    parser.add_argument("-d", "--devices", type=int, required=True,
                        help="Number of devices to use")
    parser.add_argument("-s", "--streams", type=int, required=True,
                        help="Number of streams per device")
    args = parser.parse_args()

    num_devices = args.devices
    num_streams = args.streams
    total_workers = num_devices * num_streams

    # Distribute tasks to devices
    base_tasks = args.count // num_devices
    remainder = args.count % num_devices
    device_task_counts = [base_tasks + (1 if i < remainder else 0) for i in range(num_devices)]

    # Calculate expected time
    expected = args.count * args.microseconds / 1e6 / total_workers

    print(f"Running {args.count} sleep tasks ({args.microseconds}us) with {num_devices} process(es) x {num_streams} stream(s)...")

    # Create barrier for synchronization (num_devices workers + 1 main process)
    barrier = mp.Barrier(num_devices + 1)

    # Start one process per device
    result_queue = mp.Queue()
    processes = []
    for device_id in range(num_devices):
        p = mp.Process(
            target=worker_process,
            args=(device_id, num_streams, device_task_counts[device_id], args.microseconds, result_queue, barrier)
        )
        processes.append(p)
        p.start()

    # Wait for all workers to initialize and warm up, then start timing
    barrier.wait()
    start = time.perf_counter()

    # Collect results
    total_completed = 0
    for _ in range(num_devices):
        total_completed += result_queue.get()

    for p in processes:
        p.join()

    elapsed = time.perf_counter() - start

    efficiency = expected / elapsed * 100

    print(f"Completed {args.count} sleep tasks ({args.microseconds}us each) "
          f"with {num_devices} process(es) x {num_streams} stream(s) = {total_workers} workers in {elapsed:.3f} seconds")
    print(f"Expected time (ideal): {expected:.3f}s")
    print(f"Parallel efficiency: {efficiency:.1f}%")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
