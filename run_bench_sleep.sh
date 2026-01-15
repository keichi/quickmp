#!/bin/bash
# Benchmark script to measure VEDA/VEO overhead with sleep kernel

COUNT=${1:-1000}
MICROSECONDS=${2:-5000}
MAX_STREAMS=${3:-16}

echo "=== Sleep Benchmark Parameters ==="
echo "Task count: $COUNT"
echo "Sleep duration: ${MICROSECONDS}us"
echo "Max streams: $MAX_STREAMS"
echo ""

# Calculate expected baseline
expected_baseline=$(echo "scale=3; $COUNT * $MICROSECONDS / 1000000" | bc)
echo "Expected baseline (ideal): ${expected_baseline}s"

# Run with 1 stream first to get baseline
echo "Running baseline (1 stream)..."
baseline=$(python bench_sleep.py -c $COUNT -u $MICROSECONDS -s 1 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
echo "Actual baseline time: ${baseline}s"
echo ""

echo "=== Results ==="
printf "%-10s %-15s %-10s %-15s\n" "Streams" "Time (s)" "Speedup" "Efficiency"
printf "%-10s %-15s %-10s %-15s\n" "-------" "--------" "-------" "----------"

for streams in $(seq 1 $MAX_STREAMS); do
    elapsed=$(python bench_sleep.py -c $COUNT -u $MICROSECONDS -s $streams 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
    speedup=$(echo "scale=2; $baseline / $elapsed" | bc)
    ideal_time=$(echo "scale=3; $expected_baseline / $streams" | bc)
    efficiency=$(echo "scale=1; $ideal_time / $elapsed * 100" | bc)
    printf "%-10d %-15.3f %-10.2fx %-15.1f%%\n" $streams $elapsed $speedup $efficiency
done
