#!/usr/bin/env python3
"""Summarize completed request timings. No inference or GPU benchmarking here.

CSV required: frame_id,ready_ns,submit_ns,done_ns (same steady_clock origin).
ready: BGR input available, BEFORE slot/backpressure wait.
submit: CPU starts submitting request. done: CPU has consumed final NMS result.
Measured runs exclude warmup; throughput INCLUDES measured fill and drain.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(path, expected_count=None):
    rows = []
    seen = set()
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'frame_id', 'ready_ns', 'submit_ns', 'done_ns'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('missing required CSV columns')
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('duplicate CSV columns')
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f'line {line}: extra fields')
            try:
                frame_id, ready, submit, done = [int(row[key]) for key in
                    ('frame_id', 'ready_ns', 'submit_ns', 'done_ns')]
            except (ValueError, TypeError) as exc:
                raise ValueError(f'line {line}: invalid integer') from exc
            if frame_id < 0 or frame_id in seen:
                raise ValueError(f'line {line}: negative or duplicate frame_id')
            if not 0 <= ready <= submit <= done:
                raise ValueError(f'line {line}: timestamps must satisfy 0 <= ready <= submit <= done')
            seen.add(frame_id)
            rows.append((frame_id, ready, submit, done))
    if not rows:
        raise ValueError('no completed measured frames')
    if expected_count is not None:
        if expected_count <= 0:
            raise ValueError('expected-count must be positive')
        if len(rows) != expected_count or seen != set(range(expected_count)):
            raise ValueError('incomplete/unexpected IDs: expected measured frame_id 0..N-1')
    start = min(row[1] for row in rows)
    end = max(row[3] for row in rows)
    if end <= start:
        raise ValueError('non-positive measurement window')
    latency = [(row[3] - row[1]) / 1e6 for row in rows]
    slot_wait = [(row[2] - row[1]) / 1e6 for row in rows]
    window_s = (end - start) / 1e9
    return {
        'scope': 'pipeline_e2e_from_ready_includes_slot_wait_and_measured_fill_drain',
        'frames': len(rows), 'window_s': window_s, 'fps': len(rows) / window_s,
        'latency_mean_ms': sum(latency) / len(latency),
        'latency_p50_ms': percentile(latency, 50),
        'latency_p95_ms': percentile(latency, 95),
        'latency_p99_ms': percentile(latency, 99),
        'slot_wait_mean_ms': sum(slot_wait) / len(slot_wait),
        'slot_wait_p95_ms': percentile(slot_wait, 95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--expected-count', type=int, help='Require IDs 0..N-1, catching lost/drain frames')
    parser.add_argument('--label', default='run')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.csv, args.expected_count)
        result['label'] = args.label
        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f"{args.label}: frames={result['frames']} window={result['window_s']:.6f}s FPS={result['fps']:.3f}")
        print(f"pipeline e2e ms: mean={result['latency_mean_ms']:.3f} p50={result['latency_p50_ms']:.3f} "
              f"p95={result['latency_p95_ms']:.3f} p99={result['latency_p99_ms']:.3f}")
        print(f"slot wait ms: mean={result['slot_wait_mean_ms']:.3f} p95={result['slot_wait_p95_ms']:.3f}")
        print('PASS: timing data valid. This is NOT a correctness or performance-gain verdict.')
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print(f'BENCH ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
