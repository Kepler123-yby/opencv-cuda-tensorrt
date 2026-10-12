# Module 4 验证记录

**日期**：2026-10-10
**范围**：从课文提取的预处理/解码 kernel、CPU NMS、OpenCV launcher 编译、张量比较脚本。
**不包含**：真实 TensorRT engine 推理、全流水线异步压力测试、业务模型精度或端到端性能。

## 环境

- GPU：NVIDIA GeForce RTX 3060 Ti，sm_86
- 驱动：580.178.04
- 编译器：nvcc 11.8.89、g++ 11.4.0，CUDA 代码使用 C++17
- OpenCV：4.13.0（Python CPU 参考显示 4.13.0-dev）
- NumPy：1.21.5
- TensorRT 课程目标：10.10.0.31；本轮独立 kernel 测试不加载 TensorRT

## 可复现命令

在课程根目录运行：

```bash
python3 assets/code-examples/tests/verify-module4.py \
  --arch sm_86 \
  --opencv-include "$HOME/opt/opencv-cuda/include/opencv4" \
  --memcheck
```

脚本打印临时编译目录，目录内保存提取的 `.cu`、测试程序、GPU 输出与 `metrics.json`。`--build-dir` 可指定保存位置；`--arch` 按 GPU 与 Toolkit 修改。

## 实测结果

| 检查 | 结果 |
|---|---|
| 预处理 kernel 真机运行 | 7/7 通过 |
| 所有输入使用带 pitch 的设备 ROI | 通过 |
| 1×1、横/竖图、上下/左右奇数 padding | 通过 |
| 对比 CPU float32 resize/padding | 最大绝对误差约 1.79e-7 |
| 对比 CPU uint8 resize/padding | 最大绝对误差约 0.003173，低于本测试的 2/255 阈值 |
| 解码人工测试 | 7/7 通过：两种布局、容量不足、全过滤、NaN、负宽、等于阈值 |
| 解码 ID 集合、类别、分数与四个框坐标 | 与人工参考一致 |
| class-aware NMS | 两种布局均保留 ID `[0, 2, 3]` |
| OpenCV GpuMat launcher 片段 | 编译通过；不代表完整推理工程已链接/运行 |
| Python 比较脚本 | 7/7 通过：相等、容差、失配、NaN、长度、Inf、非法 shape |
| compute-sanitizer memcheck | `ERROR SUMMARY: 0 errors` |

## 修正与边界

- 按 `new_w/new_h` 判断有效区域，避免假设 padding 对称。
- `GpuMat::step` 使用 size_t，设备 ROI 测试覆盖非连续行步长。
- 教学基线用 double 计算采样坐标，减少高分辨率输入的坐标舍入误差；后续 float 优化必须重测误差和性能。
- 原子压缩不保证顺序；比较按候选 ID 对齐，NMS 用稳定的分数/ID 排序规则。
- 检测到 capacity 溢出，不把到达次序当成 top-K。

这些结果只对本组人工样本与契约成立。真实模型的预处理契约、动态输出分配、FP16 IO、跨流生命周期、连续多帧/多 context 正确性和性能仍需按 4.4 的测试清单验证。完整独立 `.cu` 示例与 capstone 尚未完成，不能据此把全部代码素材标为已收齐。
