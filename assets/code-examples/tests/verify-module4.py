"""Extract the lesson's CUDA snippets and test them with synthetic data.

Requires nvcc, a CUDA GPU, NumPy and CPU OpenCV. Does not load a TensorRT
engine and does not measure end-to-end inference performance.
"""

from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arch', default='sm_86', help='GPU architecture for nvcc')
parser.add_argument('--opencv-include', type=Path, help='Optional OpenCV include/opencv4 path; also compile launcher')
parser.add_argument('--build-dir', type=Path, help='Keep generated sources, dumps and metrics in this directory')
parser.add_argument('--memcheck', action='store_true', help='Also run compute-sanitizer memcheck')
args = parser.parse_args()
COURSE = Path(__file__).resolve().parents[3]
ROOT = args.build_dir.resolve() if args.build_dir else Path(tempfile.mkdtemp(prefix='coach-module4-'))
ROOT.mkdir(parents=True, exist_ok=True)
print(f'Build directory: {ROOT}', flush=True)


def blocks(path):
    return re.findall(r'```cpp\n(.*?)```', path.read_text(), re.S)


pre = blocks(COURSE / 'lessons/module-4/4.1-fused-preprocess-kernel.md')
dec = blocks(COURSE / 'lessons/module-4/4.2-parallel-decode-kernel.md')
# Compile the documented kernels, geometry, and CPU NMS without modifying them.
common = pre[0] + '\n' + pre[2] + '\n' + dec[1] + '\n' + dec[3]
helper = r'''
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void dump(const std::string& name, const void* data, size_t bytes) {
    std::ofstream f(name, std::ios::binary);
    f.write(static_cast<const char*>(data), bytes);
    if (!f) throw std::runtime_error("dump failed");
}
void testPreprocess(int sw, int sh, int tw, int th, int case_id) {
    // Device source is an ROI inside a pitched allocation.
    int full_w = sw + 2, full_h = sh + 2;
    std::vector<unsigned char> host(size_t(full_w) * full_h * 3);
    for (int y = 0; y < full_h; ++y) for (int x = 0; x < full_w; ++x) {
        host[(size_t(y) * full_w + x) * 3] = (x * 13 + y * 7) % 256;
        host[(size_t(y) * full_w + x) * 3 + 1] = (x * 3 + y * 17) % 256;
        host[(size_t(y) * full_w + x) * 3 + 2] = (x * 23 + y * 5) % 256;
    }
    unsigned char* device = nullptr;
    size_t pitch = 0;
    ck(cudaMallocPitch(reinterpret_cast<void**>(&device), &pitch, full_w * 3, full_h));
    ck(cudaMemcpy2D(device, pitch, host.data(), full_w * 3, full_w * 3, full_h, cudaMemcpyHostToDevice));
    float* output = nullptr;
    ck(cudaMalloc(reinterpret_cast<void**>(&output), size_t(tw) * th * 3 * sizeof(float)));
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Letterbox lb = makeLetterbox(sw, sh, tw, th);
    letterboxBgrToRgbNchwKernel<<<(tw * th + 255) / 256, 256, 0, stream>>>(
        device + pitch + 3, pitch, sw, sh, output, tw, th,
        lb.new_w, lb.new_h, lb.pad_x, lb.pad_y, 124.0f);
    ck(cudaPeekAtLastError());
    std::vector<float> result(size_t(tw) * th * 3);
    ck(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    dump("pre-" + std::to_string(case_id) + ".bin", result.data(), result.size() * sizeof(float));
    std::cout << "preprocess case " << case_id << " pitched ROI OK, step=" << pitch
              << " lb=" << lb.new_w << "," << lb.new_h << "," << lb.pad_x << "," << lb.pad_y << "\n";
    ck(cudaFree(output)); ck(cudaFree(device)); ck(cudaStreamDestroy(stream));
}
void testDecode(int test) {
    const int K = 5, C = 2, F = 6;
    std::vector<float> rows = {
        20,20,10,10,0.9f,0.1f,
        20,20,10,10,0.8f,0.1f,
        20,20,10,10,0.1f,0.85f,
        80,80,10,10,0.7f,0.1f,
        50,50,10,10,0.2f,0.1f
    };
    int capacity = test == 2 ? 1 : K;
    int expected_count = 4;
    if (test == 3) { for (int i = 0; i < K; ++i) rows[i*F+4] = rows[i*F+5] = 0; expected_count = 0; }
    if (test == 4) { rows[4] = std::numeric_limits<float>::quiet_NaN(); expected_count = 3; }
    if (test == 5) { rows[2] = -10; expected_count = 3; }
    if (test == 6) { rows[4*F+4] = 0.5f; expected_count = 5; }
    std::vector<float> data = rows;
    bool transposed = test == 1;
    if (transposed) for (int i=0; i<K; ++i) for (int f=0; f<F; ++f) data[f*K+i] = rows[i*F+f];
    float* d_data = nullptr; int* d_count = nullptr; Candidate* d_candidates = nullptr;
    ck(cudaMalloc(reinterpret_cast<void**>(&d_data), data.size()*sizeof(float)));
    ck(cudaMalloc(reinterpret_cast<void**>(&d_count), sizeof(int)));
    ck(cudaMalloc(reinterpret_cast<void**>(&d_candidates), capacity*sizeof(Candidate)));
    ck(cudaMemcpy(d_data, data.data(), data.size()*sizeof(float), cudaMemcpyHostToDevice));
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ck(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
    decodeKernel<<<1,256,0,stream>>>(d_data,K,C,transposed?1:F,transposed?K:1,0.5f,capacity,
        100,100,100,100,0,0,d_count,d_candidates);
    ck(cudaPeekAtLastError()); ck(cudaStreamSynchronize(stream));
    int count = 0; ck(cudaMemcpy(&count,d_count,sizeof(int),cudaMemcpyDeviceToHost));
    if (count != expected_count) throw std::runtime_error("decode count mismatch");
    if (test == 2) {
        if (count <= capacity) throw std::runtime_error("overflow not reported");
    } else {
        std::vector<Candidate> boxes(count);
        if (count) ck(cudaMemcpy(boxes.data(),d_candidates,count*sizeof(Candidate),cudaMemcpyDeviceToHost));
        std::sort(boxes.begin(),boxes.end(),[](const Candidate&a,const Candidate&b){return a.candidate_id<b.candidate_id;});
        std::vector<int> expected_ids = test == 3 ? std::vector<int>{} :
            (test == 4 || test == 5) ? std::vector<int>{1,2,3} :
            test == 6 ? std::vector<int>{0,1,2,3,4} : std::vector<int>{0,1,2,3};
        std::vector<int> actual_ids;
        for (const auto& box: boxes) actual_ids.push_back(box.candidate_id);
        if (actual_ids != expected_ids) throw std::runtime_error("decode ID set mismatch");
        for (const auto& box: boxes) {
            int id=box.candidate_id;
            int cls=rows[id*F+4] >= rows[id*F+5] ? 0 : 1;
            float score=rows[id*F+4+cls];
            if (box.class_id!=cls || fabsf(box.score-score)>1e-6f ||
                fabsf(box.x1-(rows[id*F]-5))>1e-6f || fabsf(box.y1-(rows[id*F+1]-5))>1e-6f ||
                fabsf(box.x2-(rows[id*F]+5))>1e-6f || fabsf(box.y2-(rows[id*F+1]+5))>1e-6f)
                throw std::runtime_error("decode field mismatch");
        }
        if (test <= 1) {
            std::vector<int> ids;
            for (const auto& box: classAwareNms(boxes,0.5f)) ids.push_back(box.candidate_id);
            if (ids != std::vector<int>{0,2,3}) throw std::runtime_error("NMS ID mismatch");
        }
    }
    std::cout << "decode case " << test << " OK count=" << count << "\n";
    ck(cudaFree(d_data)); ck(cudaFree(d_count)); ck(cudaFree(d_candidates)); ck(cudaStreamDestroy(stream));
}
int main() {
    try {
        int sizes[][4]={{1,1,8,8},{7,5,8,8},{641,480,640,640},{480,641,640,640},
                       {1920,1080,640,640},{5,7,9,11},{7,2,13,10}};
        for (int i=0; i<7; ++i) testPreprocess(sizes[i][0],sizes[i][1],sizes[i][2],sizes[i][3],i);
        for (int i=0; i<7; ++i) testDecode(i);
        std::cout << "ALL CUDA CHECKS PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
'''
source = ROOT / 'module4-test.cu'
source.write_text(common + helper)
cmd = ['nvcc', '-std=c++17', f'-arch={args.arch}', '-lineinfo', str(source), '-o', str(ROOT / 'module4-test')]
subprocess.run(cmd, check=True, cwd=ROOT)
subprocess.run([str(ROOT / 'module4-test')], check=True, cwd=ROOT)

# Optionally compile the launcher; this is not a TensorRT pipeline run.
if args.opencv_include:
    launcher = ROOT / 'launcher-check.cu'
    launcher.write_text(pre[0] + '\n' + pre[2] + '\n' + pre[3])
    subprocess.run(['nvcc', '-std=c++17', f'-arch={args.arch}', f'-I{args.opencv_include}',
                    '-c', str(launcher), '-o', str(ROOT / 'launcher.o')], check=True, cwd=ROOT)

shapes = [(1,1,8,8),(7,5,8,8),(641,480,640,640),(480,641,640,640),(1920,1080,640,640),(5,7,9,11),(7,2,13,10)]
metrics = []
for i,(sw,sh,tw,th) in enumerate(shapes):
    y,x = np.mgrid[:sh+2,:sw+2]
    full = np.stack([(x*13+y*7)%256,(x*3+y*17)%256,(x*23+y*5)%256],axis=-1).astype(np.uint8)
    image = full[1:sh+1,1:sw+1]
    scale=min(tw/sw,th/sh)
    nw=min(tw,max(1,int(np.floor(sw*scale+0.5))))
    nh=min(th,max(1,int(np.floor(sh*scale+0.5))))
    px=(tw-nw)//2; py=(th-nh)//2
    def reference(dtype):
        resized=cv2.resize(image.astype(dtype),(nw,nh),interpolation=cv2.INTER_LINEAR)
        padded=cv2.copyMakeBorder(resized,py,th-nh-py,px,tw-nw-px,cv2.BORDER_CONSTANT,value=(124,124,124))
        return padded[:,:,::-1].transpose(2,0,1).astype(np.float32)/255.0
    gpu=np.fromfile(ROOT / f'pre-{i}.bin',dtype='<f4').reshape(3,th,tw)
    float_error=float(np.max(np.abs(gpu-reference(np.float32))))
    uint8_error=float(np.max(np.abs(gpu-reference(np.uint8))))
    if float_error>1e-5 or uint8_error>2/255:
        raise AssertionError(f'preprocess parity failed case {i}: {float_error}, {uint8_error}')
    metrics.append(dict(case=i,source=[sw,sh],target=[tw,th],float_max_error=float_error,uint8_max_error=uint8_error))
    print(f'preprocess parity case {i}: float max={float_error:.8g}; uint8 max={uint8_error:.8g} PASS')

# Exercise the actual CLI script: exact match, tolerances, mismatch, NaN, length, invalid shape.
compare=COURSE / 'assets/code-examples/4.4-compare-tensors.py'
a=np.array([0,0.5,1],dtype='<f4')
a.tofile(ROOT/'ref.bin')
script_tests=[('exact',a,0,['--atol','0']),('tolerance',a+0.0001,0,['--atol','0.001']),
              ('mismatch',a+0.1,1,['--atol','0.001']),('nan',np.array([0,np.nan,1],dtype='<f4'),2,[]),
              ('length',a[:2],2,[]),('inf',np.array([0,np.inf,1],dtype='<f4'),2,[])]
for name,arr,expected,extra in script_tests:
    arr.tofile(ROOT/'actual.bin')
    result=subprocess.run([sys.executable,str(compare),str(ROOT/'ref.bin'),str(ROOT/'actual.bin'),
                           '--shape','1','3']+extra,capture_output=True,text=True)
    if result.returncode!=expected: raise AssertionError(name+' wrong exit code: '+result.stdout+result.stderr)
    print(f'comparison CLI {name}: exit={expected} PASS')
result=subprocess.run([sys.executable,str(compare),str(ROOT/'ref.bin'),str(ROOT/'ref.bin'),
                       '--shape','0','3'],capture_output=True,text=True)
if result.returncode!=2: raise AssertionError('zero shape accepted')
print('comparison CLI invalid shape: exit=2 PASS')
(ROOT/'metrics.json').write_text(json.dumps(metrics,indent=2))
if args.memcheck:
    subprocess.run(['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                    str(ROOT / 'module4-test')], check=True, cwd=ROOT)
print('VERIFIED: CUDA kernels (7+7 cases), Python CLI (7 cases).')
if args.opencv_include:
    print('VERIFIED: OpenCV launcher compiled; TensorRT pipeline not executed.')
