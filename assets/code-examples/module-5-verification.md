# Module 5 验证记录

日期：2026-10-10。三节课时已出稿，验证仅覆盖辅助工具与 C++ 脚手架，不代表完整 detector 已实现或已完成实测。

## 环境

- g++ 11.4.0、C++20；Python 3.10.12；yaml-cpp 0.7.0。
- OpenCV 头文件：`/home/kepler/opt/opencv-cuda/include/opencv4`。
- TensorRT 头文件：`/home/kepler/tensorrt/TensorRT-10.10.0.31/include`。
- CUDA 头文件：`/usr/local/cuda/include`，本轮只用于语法检查，不是运行时兼容性证明。

## 复现

课程根目录执行：

```bash
python3 assets/code-examples/tests/verify-module5.py \
  --opencv-include "$HOME/opt/opencv-cuda/include/opencv4" \
  --trt-include "$HOME/tensorrt/TensorRT-10.10.0.31/include" \
  --cuda-include /usr/local/cuda/include
```

输出：

```text
PASS: {"benchmark": 15, "config": 26, "cpp_syntax": 3}
Scope: CPU tool tests and optional compile-only checks; NO GPU inference/performance validation.
```

不提供头文件参数时，仍执行配置/统计工具测试，但明确跳过 C++ 语法检查。所有临时 YAML、CSV、JSON、二进制放入临时目录并在退出时清理。

## 验证覆盖

| 类别 | 通过 | 说明 |
|---|---:|---|
| YAML 配置 | 26 | 8 个合法组合；18 个非法配置，覆盖 CUDA 许可、mode/slots、枚举、shape、范围、未知/重复键以及路径/IO 名称 null |
| CSV 统计 | 15 | 串行人工数据与乱序并行人工数据；FPS、均值/分位数和背压等待核对；13 个非法/不完整输入拒绝 |
| C++ 语法 | 3 | resolved shape 字节数检查、含背压/drain 的调度片段、抽象 Pipeline 接口；仅 `-fsyntax-only` |

统计基准完全由人工时间戳构造，不使用推理结果或 GPU 计时。乱序样本 FPS=150，mean latency=13.333… ms，证明脚本未用单帧均值倒数替代多路吞吐。

只读审查发现 yaml-cpp 可将 null 转成字符串 `"null"`；已在 text() 转换前强制非 null scalar，并加入 onnx/engine/input_name/output_name 四项回归测试。

17 个课时文件齐全，Module 5 的相对资产链接全部可解析。

## 未验证/仍需学员完成

- 完整 CMake 工程与 detector 后端实现、ONNX/engine/video 集成；5.3 提供的是实现规范和接口脚手架。
- TensorRT engine 构建/加载、插件及运行时兼容性、真实模型 IO 语义。
- GPU 全流水线、槽位竞争/错帧、异常回收、压力测试、完整工程 memcheck。
- A–E 的真实端到端延迟/吞吐、Nsight 时间线和资源峰值。

Module 4 的独立 kernel 真机测试是另一份记录；不得据此或本轮语法检查声称完整推理工程已验收。
