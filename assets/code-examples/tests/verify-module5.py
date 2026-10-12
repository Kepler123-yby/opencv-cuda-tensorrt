#!/usr/bin/env python3
"""Exercise Module 5 CPU tools; optionally syntax-check C++ scaffolds.

No ONNX/engine loading, GPU inference, race testing, or real performance claims.
Requires g++, pkg-config and yaml-cpp. Optional checks need OpenCV/TRT headers.
All generated inputs/binaries live in a temporary directory.
"""
import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

COURSE = Path(__file__).resolve().parents[3]
ASSETS = COURSE / 'assets/code-examples'
COUNTS = {'config': 0, 'benchmark': 0, 'cpp_syntax': 0}


def run(command, expected=0, contains=None):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode != expected or (contains and contains not in output):
        raise RuntimeError(f'command {command}\nexit={result.returncode}; expected={expected}\n{output}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--opencv-include', type=Path)
    parser.add_argument('--trt-include', type=Path)
    parser.add_argument('--cuda-include', type=Path)
    args = parser.parse_args()
    if bool(args.opencv_include) != bool(args.trt_include):
        parser.error('provide both --opencv-include and --trt-include for syntax checks')
    with tempfile.TemporaryDirectory(prefix='coach-module5-') as directory:
        tmp = Path(directory)
        flags = shlex.split(run(['pkg-config', '--cflags', '--libs', 'yaml-cpp']))
        validator = tmp / 'validate-config'
        run([args.cxx, '-std=c++20', '-Wall', '-Wextra', ASSETS / '5.1-validate-config.cpp',
             *flags, '-o', validator])
        base = (COURSE / 'assets/configs/5.1-deploy.yaml').read_text()

        def change(text, old, new):
            if text.count(old) != 1:
                raise RuntimeError(f'fixture replacement must be unique: {old}')
            return text.replace(old, new)

        def mutate(*pairs):
            text = base
            for old, new in pairs:
                text = change(text, old, new)
            return text

        cpu = mutate(('backend: tensorrt', 'backend: opencv_cpu'),
                     ('cuda_enabled: true', 'cuda_enabled: false'),
                     ('implementation: fused_cuda', 'implementation: cpu'),
                     ('implementation: cuda_decode', 'implementation: cpu'))
        dnn_cuda = change(change(cpu, 'backend: opencv_cpu', 'backend: opencv_cuda'),
                          'cuda_enabled: false', 'cuda_enabled: true')
        valid = [base, cpu, dnn_cuda,
                 mutate(('implementation: fused_cuda', 'implementation: cpu'),
                        ('implementation: cuda_decode', 'implementation: cpu')),
                 mutate(('mode: sync', 'mode: parallel'), ('slots: 1', 'slots: 2')),
                 mutate(('mode: sync', 'mode: parallel'), ('slots: 1', 'slots: 4')),
                 mutate(('mode: sync', 'mode: async')),
                 mutate(('precision: fp32', 'precision: fp16'))]
        invalid = [
            (mutate(('cuda_enabled: true', 'cuda_enabled: false')), 'GPU path requested'),
            (mutate(('mode: sync', 'mode: parallel')), 'mode/slots mismatch'),
            (mutate(('slots: 1', 'slots: 9')), 'slots must be'),
            (change(cpu, 'mode: sync', 'mode: async'), 'only for TensorRT'),
            (change(cpu, 'precision: fp32', 'precision: fp16'), 'supports fp32 only'),
            (mutate(('backend: tensorrt', 'backend: typo')), 'backend: invalid value'),
            (mutate(('input_shape: [1, 3, 640, 640]', 'input_shape: [1, 3, -1, 640]')), 'positive [1,3,H,W]'),
            (mutate(('pad_value: 124', 'pad_value: 12.5')), 'pad_value must be'),
            (mutate(('score_threshold: 0.5', 'score_threshold: .nan')), 'must be finite'),
            (mutate(('num_classes: 2', 'num_classes: 0')), 'num_classes must be'),
            (mutate(('warmup: 30', 'warmup: -1')), 'warmup must be'),
            (mutate(('frames: 300', 'frames: 0')), 'frames must be'),
            (base + 'typo: true\n', 'unknown key'),
            (base + 'schema_version: 1\n', 'duplicate key'),
        ]
        # Regression: yaml-cpp converts null to the string "null" without a type check.
        for key, value in [('onnx', '../models/detector.onnx'),
                           ('engine', '../cache/detector-trt10.10-fp32.engine'),
                           ('input_name', 'images'), ('output_name', 'output0')]:
            invalid.append((change(base, f'{key}: {value}', f'{key}:'), 'non-null scalar field'))
        for text in valid:
            path = tmp / 'config.yaml'
            path.write_text(text)
            run([validator, path], contains='PASS: schema')
            COUNTS['config'] += 1
        for text, message in invalid:
            path = tmp / 'config.yaml'
            path.write_text(text)
            run([validator, path], expected=2, contains=message)
            COUNTS['config'] += 1

        stats = ASSETS / '5.2-summarize-benchmark.py'
        header = 'frame_id,ready_ns,submit_ns,done_ns\n'
        csv_path, json_path = tmp / 'timings.csv', tmp / 'summary.json'
        valid_csv = [
            (header + '0,0,0,10000000\n1,10000000,10000000,20000000\n2,20000000,20000000,30000000\n',
             3, {'fps': 100, 'latency_mean_ms': 10, 'latency_p99_ms': 10, 'slot_wait_mean_ms': 0}),
            # Out-of-order rows; latency is not the inverse of throughput.
            (header + '2,2000000,8000000,20000000\n0,0,0,10000000\n1,1000000,6000000,13000000\n',
             3, {'fps': 150, 'latency_mean_ms': 40/3, 'latency_p50_ms': 12,
                 'latency_p95_ms': 17.4, 'latency_p99_ms': 17.88, 'slot_wait_mean_ms': 11/3}),
        ]
        for text, count, expected in valid_csv:
            csv_path.write_text(text)
            run([sys.executable, stats, csv_path, '--expected-count', count,
                 '--json-out', json_path], contains='PASS: timing data valid')
            result = json.loads(json_path.read_text())
            for key, value in expected.items():
                if not math.isclose(result[key], value, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError(f'{key}: {result[key]} != {value}')
            COUNTS['benchmark'] += 1
        bad_csv = [
            (header, 1, 'no completed'),
            ('frame_id,ready_ns,submit_ns\n0,0,0\n', 1, 'missing required'),
            (header + '0,0,0,100\n0,0,0,200\n', 2, 'duplicate frame_id'),
            (header + '0,0,0,100\n2,0,0,200\n', 2, 'incomplete/unexpected IDs'),
            (header + '0,10,0,100\n', 1, 'timestamps must satisfy'),
            (header + '0,0,100,10\n', 1, 'timestamps must satisfy'),
            (header + '0,-1,0,100\n', 1, 'timestamps must satisfy'),
            (header + '0,0,0,0\n', 1, 'non-positive measurement window'),
            (header + '-1,0,0,100\n', 1, 'negative or duplicate'),
            (header + '0,0,0,1.5\n', 1, 'invalid integer'),
            (header + '0,0,0,100,extra\n', 1, 'extra fields'),
            (header.replace('done_ns', 'done_ns,done_ns') + '0,0,0,100,100\n', 1, 'duplicate CSV'),
            (header + '0,0,0,100\n', 0, 'expected-count must be positive'),
        ]
        for text, count, message in bad_csv:
            csv_path.write_text(text)
            run([sys.executable, stats, csv_path, '--expected-count', count], 2, message)
            COUNTS['benchmark'] += 1

        if args.opencv_include:
            (tmp / 'pipeline_contract.hpp').write_text((ASSETS / '5.3-pipeline-contract.hpp').read_text())
            syntax_flags = [args.cxx, '-std=c++20', '-fsyntax-only', '-I', tmp,
                            '-I', args.opencv_include, '-I', args.trt_include]
            if args.cuda_include:
                syntax_flags += ['-I', args.cuda_include]
            lesson = (COURSE / 'lessons/module-5/5.3-capstone-project.md').read_text()
            snippets = re.findall(r'```cpp\n(.*?)\n```', lesson, re.S)
            snippets = [s for s in snippets if '#include <NvInferRuntime.h>' in s or
                        '#include "pipeline_contract.hpp"' in s]
            if len(snippets) != 2:
                raise RuntimeError('expected shape-check and scheduler snippets')
            snippets.append('#include "pipeline_contract.hpp"\n#include <type_traits>\n'
                            'static_assert(std::is_abstract_v<coach::Pipeline>);\n')
            for i, snippet in enumerate(snippets):
                source = tmp / f'syntax-{i}.cpp'
                source.write_text(snippet)
                run([*syntax_flags, source])
                COUNTS['cpp_syntax'] += 1
        else:
            print('SKIPPED: C++ scaffold syntax (provide header paths to enable)')
        print('PASS:', json.dumps(COUNTS, sort_keys=True))
        print('Scope: CPU tool tests and optional compile-only checks; NO GPU inference/performance validation.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'VERIFY ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
