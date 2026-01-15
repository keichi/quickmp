#!/bin/bash
# Benchmark script to measure speedup with varying number of streams

COUNT=${1:-1000}
LENGTH=${2:-7200}
WINDOW=${3:-10}
MAX_STREAMS=${4:-16}

echo "=== Benchmark Parameters ==="
echo "Time series count: $COUNT"
echo "Time series length: $LENGTH"
echo "Window size: $WINDOW"
echo "Max streams: $MAX_STREAMS"
echo ""

# Run with 1 stream first to get baseline
echo "Running baseline (1 stream)..."
baseline=$(python bench.py -c $COUNT -n $LENGTH -m $WINDOW -s 1 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
echo "Baseline time: ${baseline}s"
echo ""

echo "=== Results ==="
printf "%-10s %-15s %-10s\n" "Streams" "Time (s)" "Speedup"
printf "%-10s %-15s %-10s\n" "-------" "--------" "-------"

for streams in $(seq 1 $MAX_STREAMS); do
    elapsed=$(python bench.py -c $COUNT -n $LENGTH -m $WINDOW -s $streams 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
    speedup=$(echo "scale=2; $baseline / $elapsed" | bc)
    printf "%-10d %-15.3f %-10.2fx\n" $streams $elapsed $speedup
done
