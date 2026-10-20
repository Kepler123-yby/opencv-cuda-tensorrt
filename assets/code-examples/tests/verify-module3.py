#!/usr/bin/env python3
"""Check the actual Module 3 scheduler with CPU FakeSlots; requires g++.

Extracts functions from 3.3-infer-modes.cpp, compiles in a temporary directory,
and checks scheduling/percentile contracts. No CUDA, engine, or GPU execution.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1]

PREFIX = r'''
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
struct IOInfo { size_t in_elements = 17; size_t out_elements = 3; };
void prepareInput(uint64_t, std::vector<float>&);
struct Slot {
    bool active = false;
    uint64_t frame_id = 0;
    size_t submitted = 0, collected = 0;
    bool busy() const noexcept { return active; }
    uint64_t frameId() const noexcept { return frame_id; }
    void submit(uint64_t id, const std::vector<float>& input) {
        if (active) throw std::runtime_error("slot overwritten before collect");
        std::vector<float> expected(input.size());
        prepareInput(id, expected);
        if (expected != input) throw std::runtime_error("wrong input buffer");
        frame_id = id; active = true; ++submitted;
    }
    bool ready() const {
        if (!active) throw std::runtime_error("query idle slot");
        return false; // collect must still work when a readiness query says not ready.
    }
    void collect(uint64_t expected, std::vector<float>& output) {
        if (!active) throw std::runtime_error("double collect");
        if (expected != frame_id) throw std::runtime_error("wrong collect id");
        output = {static_cast<float>(frame_id), 1.f, 2.f};
        active = false; ++collected;
    }
};
'''

MAIN = r'''
int main() {
    size_t cases = 0, outputs = 0;
    IOInfo io;
    for (Mode mode : {Mode::Sync, Mode::Async, Mode::Parallel}) {
        for (size_t lanes = 1; lanes <= 19; ++lanes) {
            for (size_t frames = 1; frames <= 103; ++frames) {
                const size_t count = mode == Mode::Parallel ? lanes : 1;
                std::vector<std::unique_ptr<Slot>> slots;
                for (size_t i = 0; i < count; ++i) slots.push_back(std::make_unique<Slot>());
                for (size_t repetition = 0; repetition < 3; ++repetition) {
                    size_t consumed = 0;
                    auto result = runSchedule(mode, slots, io, frames,
                        [&](uint64_t id, const std::vector<float>& output) {
                            if (id != consumed) throw std::runtime_error("not exactly once/in order");
                            if (output.size() != io.out_elements || output[0] != static_cast<float>(id))
                                throw std::runtime_error("wrong output ownership");
                            ++consumed;
                        });
                    if (consumed != frames) throw std::runtime_error("missing results/drain");
                    if (result.latency_ms.size() != frames || result.wall_seconds < 0)
                        throw std::runtime_error("invalid measurement");
                    for (const auto& slot : slots)
                        if (slot->busy() || slot->submitted != slot->collected)
                            throw std::runtime_error("undrained or unbalanced slot");
                    for (double t : result.latency_ms)
                        if (!std::isfinite(t) || t < 0) throw std::runtime_error("invalid latency");
                    ++cases; outputs += consumed;
                }
            }
        }
    }
    auto equal = [](double actual, double expected) {
        if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12)
            throw std::runtime_error("percentile must use (N-1)*p linear interpolation");
    };
    equal(percentile({7}, 0.95), 7);
    equal(percentile({1, 2, 3, 4}, 0.0), 1);
    equal(percentile({1, 2, 3, 4}, 0.5), 2.5);
    equal(percentile({1, 2, 3, 4}, 0.95), 3.85);
    equal(percentile({1, 2, 3, 4}, 0.99), 3.97);
    equal(percentile({1, 2, 3, 4}, 1.0), 4);
    std::cout << "PASS: CPU scheduler protocol, cases=" << cases << " outputs=" << outputs
              << "; ordered exactly-once collection, safe slot reuse, correct inputs, full drain, "
                 "repeat runs; percentile=6 checks.\n"
              << "Scope: extracted scheduler with FakeSlots; NO GPU inference, concurrency, "
                 "CUDA error cleanup, or performance validation.\n";
}
'''


def section(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f'expected unique source markers: {start!r}, {end!r}')
    return source[source.index(start):source.index(end)]


def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'command {command}\nexit={result.returncode}\n'
                           f'{result.stdout}{result.stderr}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    source = (ASSETS / '3.3-infer-modes.cpp').read_text()
    code = (PREFIX + section(source, 'void prepareInput(', 'void warmup(')
            + section(source, 'enum class Mode', 'void runMode(') + MAIN)
    with tempfile.TemporaryDirectory(prefix='coach-module3-') as directory:
        cpp, binary = Path(directory) / 'scheduler.cpp', Path(directory) / 'scheduler'
        cpp.write_text(code)
        run([args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-pedantic', cpp, '-o', binary])
        print(run([binary]), end='')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
