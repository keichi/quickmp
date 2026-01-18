#!/bin/bash
# Benchmark script to measure speedup with varying number of devices and streams

COUNT=${1:-1000}
LENGTH=${2:-7200}
WINDOW=${3:-10}

DEVICES=(1 2 4 8)
STREAMS=(1 2 4 8 16)

echo "=== Benchmark Parameters ==="
echo "Time series count: $COUNT"
echo "Time series length: $LENGTH"
echo "Window size: $WINDOW"
echo "Devices: ${DEVICES[*]}"
echo "Streams: ${STREAMS[*]}"
echo ""

# Run with 1 device, 1 stream first to get baseline
echo "Running baseline (1 device, 1 stream)..."
baseline=$(python bench.py -c $COUNT -n $LENGTH -m $WINDOW -d 1 -s 1 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
echo "Baseline time: ${baseline}s"
echo ""

echo "=== Results ==="
printf "%-10s %-10s %-10s %-15s %-10s\n" "Devices" "Streams" "Workers" "Time (s)" "Speedup"
printf "%-10s %-10s %-10s %-15s %-10s\n" "-------" "-------" "-------" "--------" "-------"

for devices in "${DEVICES[@]}"; do
    for streams in "${STREAMS[@]}"; do
        workers=$((devices * streams))
        elapsed=$(python bench.py -c $COUNT -n $LENGTH -m $WINDOW -d $devices -s $streams 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
        if [ -n "$elapsed" ]; then
            speedup=$(echo "scale=2; $baseline / $elapsed" | bc)
            printf "%-10d %-10d %-10d %-15.3f %-10.2fx\n" $devices $streams $workers $elapsed $speedup
        else
            printf "%-10d %-10d %-10d %-15s %-10s\n" $devices $streams $workers "ERROR" "-"
        fi
    done
done
