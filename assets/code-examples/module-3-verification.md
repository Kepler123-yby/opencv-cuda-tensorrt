# Module 3.3 示例验证记录

验证日期：2026-10-15。对象：[`3.3-infer-modes.cpp`](3.3-infer-modes.cpp) 与 [3.3 课文](../../lessons/module-3/3.3-sync-async-parallel.md)。本记录只覆盖本轮新增的 3.3 素材，不代替 3.1 / 3.2 的验证记录。

## 环境与范围

- g++ 11.4.0，C++17。
- TensorRT 10.10.0.31：`/home/kepler/tensorrt/TensorRT-10.10.0.31`。
- CUDA Toolkit 11.8.89：`/home/kepler/cuda-11.8`。
- 当前 `nvidia-smi -L` 无法连接 NVIDIA 驱动，未发现 `/dev/nvidia*`。

已完成编译链接、CLI 负向检查和不依赖 CUDA 的 CPU 调度协议检查。**未执行真实 Engine 推理、GPU 输出对比、异步执行错误注入、并发时间线或性能实测**。源码中的正确性检查会在有 GPU 的环境运行时执行；写有检查逻辑不等于本轮已通过真机验证。

## 编译与 CLI 检查

在仓库根目录执行：

```bash
export TRT_ROOT="$HOME/tensorrt/TensorRT-10.10.0.31"
export CUDA_ROOT="$HOME/cuda-11.8"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    assets/code-examples/3.3-infer-modes.cpp -o /tmp/infer-modes \
    -isystem "$TRT_ROOT/include" -isystem "$CUDA_ROOT/include" \
    -L"$TRT_ROOT/lib" -L"$CUDA_ROOT/lib64" -lnvinfer -lcudart
```

结果：退出码 0。普通 `-I` 版本也编译链接成功；`-isystem` 隔离供应商头文件告警后，样例在 `-Wall -Wextra -Werror` 下通过。

| 命令 | 预期 / 实际结果 |
|---|---|
| `/tmp/infer-modes` | 用法提示，退出码 1 |
| `/tmp/infer-modes /tmp/no-such-3.3.engine` | 无法打开 Engine，退出码 1 |
| `/tmp/infer-modes /tmp/no-such-3.3.engine 0 3` | 拒绝零帧数，退出码 1 |
| `/tmp/infer-modes /tmp/no-such-3.3.engine 7 0` | 拒绝零路数，退出码 1 |
| `/tmp/infer-modes /tmp/no-such-3.3.engine 7x 3` | 拒绝非法数字，退出码 1 |

这些检查在 Engine 反序列化之前结束，不需要驱动，也不能验证模型或 GPU 路径。

## CPU 调度协议检查

```bash
python3 assets/code-examples/tests/verify-module3.py
```

脚本从当前源文件提取 `prepareInput`、`Mode / Measurement`、`runSchedule` 和分位数实现，将 GPU Slot 替换成 CPU FakeSlot，然后编译执行。测试不保留另一份固定的调度实现。

覆盖 sync / async / parallel、lanes=1..19、frames=1..103，每种组合连续运行 3 次，共 **17,613 次调度、915,876 个输出**。检查：

- 每帧结果按序且恰好消费一次，输入对应正确的 frame_id。
- 不覆盖在途槽位，不重复 collect，不以错误编号回收。
- 帧数小于槽位数、非整除槽位数、多次绕环都完整排空。
- 全部槽位在排空后为空闲，能够再次使用。
- 返回的延迟数量匹配帧数、时间非负；分位数采用与 5.2 相同的线性插值。

结果：通过，另有 6 项分位数边界 / 插值检查通过。FakeSlot 只验证 CPU 调度契约，**不模拟 CUDA stream/event、真实内存传输、Context 并发或 GPU 耗时**。时间数值不作为性能数据。

## 有 GPU 后仍需完成

按课文命令运行真实静态 Engine，先通过同步参考与三种模式的逐元素对比，再检查 frames / lanes 边界和 Nsight Systems 时间线。记录 GPU、驱动、模型与 Engine、输入、精度、路数、正确结果数、mean / p50 / p95、FPS 和显存。无实测前不填写性能数字，也不预设并行一定更快。
