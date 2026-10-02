#!/bin/bash
# Benchmark script to measure speedup with varying number of devices

COUNT=${1:-1000}
LENGTH=${2:-7200}
WINDOW=${3:-10}

DEVICES=(1 2 4 8)

echo "=== Benchmark Parameters ==="
echo "Time series count: $COUNT"
echo "Time series length: $LENGTH"
echo "Window size: $WINDOW"
echo "Devices: ${DEVICES[*]}"
echo ""

echo "=== Results ==="
printf "%-10s %-15s %-15s\n" "Devices" "Time (s)" "Speedup"
printf "%-10s %-15s %-15s\n" "-------" "--------" "-------"

baseline=""
for devices in "${DEVICES[@]}"; do
    elapsed=$(python bench.py -c $COUNT -n $LENGTH -m $WINDOW -d $devices 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
    if [ -n "$elapsed" ]; then
        baseline=${baseline:-$elapsed}
        speedup=$(echo "scale=2; $baseline / $elapsed" | bc)
        printf "%-10d %-15.3f %-15.2fx\n" $devices $elapsed $speedup
    else
        printf "%-10d %-15s %-15s\n" $devices "ERROR" "-"
    fi
done
