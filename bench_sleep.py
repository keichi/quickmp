#!/usr/bin/env python3
"""Benchmark to measure VEDA/VEO overhead using sleep kernel."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import quickmp


def run_sleep(args):
    """Run sleep on VE."""
    microseconds, stream = args
    quickmp.sleep_us(microseconds, stream)


def main():
    parser = argparse.ArgumentParser(description="Sleep benchmark for quickmp")
    parser.add_argument("-c", "--count", type=int, default=1000,
                        help="Number of sleep tasks (default: 1000)")
    parser.add_argument("-u", "--microseconds", type=int, default=1000,
                        help="Sleep duration in microseconds (default: 1000)")
    parser.add_argument("-s", "--streams", type=int, default=1,
                        help="Number of streams (default: 1)")
    args = parser.parse_args()

    quickmp.initialize()

    # Warm up
    for i in range(args.streams):
        quickmp.sleep_us(100, i)

    # Prepare tasks
    tasks = [(args.microseconds, i % args.streams) for i in range(args.count)]

    # Run benchmark
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.streams) as executor:
        list(executor.map(run_sleep, tasks))
    elapsed = time.perf_counter() - start

    # Calculate expected time
    expected = args.count * args.microseconds / 1e6 / args.streams
    efficiency = expected / elapsed * 100

    print(f"Completed {args.count} sleep tasks ({args.microseconds}us each) "
          f"with {args.streams} streams in {elapsed:.3f} seconds")
    print(f"Expected time (ideal): {expected:.3f}s")
    print(f"Parallel efficiency: {efficiency:.1f}%")

    quickmp.finalize()


if __name__ == "__main__":
    main()
