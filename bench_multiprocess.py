#!/usr/bin/env python3
"""Benchmark with one process per device to avoid VEDA lock contention."""

import argparse
import multiprocessing as mp
import sys
import time


def worker_process(device_id, num_streams, tasks, window, result_queue, barrier):
    """Worker process that handles one VE device."""
    import threading
    import numpy as np
    import quickmp

    # Initialize only this device
    quickmp.initialize(device_start=device_id, device_count=1)
    quickmp.use_device(0)  # Device 0 is the only initialized device

    # Warm up
    if tasks:
        T = tasks[0][1]
        for s in range(num_streams):
            quickmp.selfjoin(T, window, stream=s)

    # Distribute tasks to streams
    stream_tasks = [[] for _ in range(num_streams)]
    for i, (idx, T) in enumerate(tasks):
        stream_id = i % num_streams
        stream_tasks[stream_id].append((idx, T))

    results = [None] * len(tasks)
    results_lock = threading.Lock()

    def stream_worker(stream_id, stream_task_list):
        quickmp.use_device(0)  # Each thread needs to set device (thread_local)
        for idx, T in stream_task_list:
            mp_result = quickmp.selfjoin(T, window, stream=stream_id)
            with results_lock:
                # Find position in original tasks list
                for i, (orig_idx, _) in enumerate(tasks):
                    if orig_idx == idx:
                        results[i] = (idx, mp_result)
                        break

    # Wait for all processes to be ready
    barrier.wait()

    # Execute with threads (one per stream)
    threads = []
    for stream_id in range(num_streams):
        if stream_tasks[stream_id]:
            t = threading.Thread(target=stream_worker, args=(stream_id, stream_tasks[stream_id]))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    quickmp.finalize()
    result_queue.put([r for r in results if r is not None])


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark quickmp with one process per device"
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
        "-d", "--devices", type=int, required=True,
        help="Number of devices to use"
    )
    parser.add_argument(
        "-s", "--streams", type=int, required=True,
        help="Number of streams per device"
    )
    args = parser.parse_args()

    import numpy as np

    print(f"Generating {args.count} time series of length {args.length}...")
    np.random.seed(42)
    timeseries_list = [np.random.rand(args.length) for _ in range(args.count)]

    num_devices = args.devices
    num_streams = args.streams
    total_workers = num_devices * num_streams

    # Distribute tasks to devices (round-robin)
    device_tasks = [[] for _ in range(num_devices)]
    for idx, T in enumerate(timeseries_list):
        device_id = idx % num_devices
        device_tasks[device_id].append((idx, T))

    print(f"Computing matrix profiles with {num_devices} process(es) x {num_streams} stream(s) = {total_workers} workers...")

    # Create barrier for synchronization (num_devices workers + 1 main process)
    barrier = mp.Barrier(num_devices + 1)

    # Start one process per device
    result_queue = mp.Queue()
    processes = []
    for device_id in range(num_devices):
        p = mp.Process(
            target=worker_process,
            args=(device_id, num_streams, device_tasks[device_id], args.window, result_queue, barrier)
        )
        processes.append(p)
        p.start()

    # Wait for all workers to initialize and warm up, then start timing
    barrier.wait()
    start = time.perf_counter()

    # Collect results
    all_results = []
    for _ in range(num_devices):
        results = result_queue.get()
        all_results.extend(results)

    for p in processes:
        p.join()

    elapsed = time.perf_counter() - start

    print(f"Completed {args.count} matrix profiles in {elapsed:.3f} seconds")
    print(f"Throughput: {args.count / elapsed:.2f} profiles/sec")
    print(f"Average time per profile: {elapsed / args.count * 1000:.3f} ms")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
