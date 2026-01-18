#!/bin/bash
# Benchmark script to measure VEDA/VEO overhead with sleep kernel (static distribution)

COUNT=${1:-1000}
MICROSECONDS=${2:-5000}

DEVICES=(1 2 4 8)
STREAMS=(1 2 4 8 16)

echo "=== Sleep Benchmark Parameters (Static) ==="
echo "Task count: $COUNT"
echo "Sleep duration: ${MICROSECONDS}us"
echo "Devices: ${DEVICES[*]}"
echo "Streams: ${STREAMS[*]}"
echo ""

# Calculate expected baseline
expected_baseline=$(echo "scale=3; $COUNT * $MICROSECONDS / 1000000" | bc)
echo "Expected baseline (ideal): ${expected_baseline}s"

# Run with 1 device, 1 stream first to get baseline
echo "Running baseline (1 device, 1 stream)..."
baseline=$(python bench_sleep_static.py -c $COUNT -u $MICROSECONDS -d 1 -s 1 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
echo "Actual baseline time: ${baseline}s"
echo ""

echo "=== Results ==="
printf "%-10s %-10s %-10s %-15s %-10s %-15s\n" "Devices" "Streams" "Workers" "Time (s)" "Speedup" "Efficiency"
printf "%-10s %-10s %-10s %-15s %-10s %-15s\n" "-------" "-------" "-------" "--------" "-------" "----------"

for devices in "${DEVICES[@]}"; do
    for streams in "${STREAMS[@]}"; do
        workers=$((devices * streams))
        elapsed=$(python bench_sleep_static.py -c $COUNT -u $MICROSECONDS -d $devices -s $streams 2>&1 | grep "Completed" | sed 's/.*in \([0-9.]*\) seconds/\1/')
        if [ -n "$elapsed" ]; then
            speedup=$(echo "scale=2; $baseline / $elapsed" | bc)
            ideal_time=$(echo "scale=3; $expected_baseline / $workers" | bc)
            efficiency=$(echo "scale=1; $ideal_time / $elapsed * 100" | bc)
            printf "%-10d %-10d %-10d %-15.3f %-10.2fx %-15.1f%%\n" $devices $streams $workers $elapsed $speedup $efficiency
        else
            printf "%-10d %-10d %-10d %-15s %-10s %-15s\n" $devices $streams $workers "ERROR" "-" "-"
        fi
    done
done
