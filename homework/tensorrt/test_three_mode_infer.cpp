// 三种推理方式的对拍测试：
//   1) 同步 infer 与 async / parallel 结果一致
//   2) 多请求并发时通道之间不串数据
//   3) 并发限流不超发线程、不死锁
//   4) Inference 提前销毁（future 未 get）不会崩
//   5) 非法配置能被正确拒绝
#include "three_mode_infer.hpp"

#include <cmath>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

bool same(const TRTInfer::Output& a, const TRTInfer::Output& b, float tol = 1e-4f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int failures = 0;

void report(const std::string& name, bool ok, const std::string& extra = "")
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
    if (!extra.empty()) std::cout << " " << extra;
    std::cout << "\n";
    if (!ok) ++failures;
}

template <typename Fn>
void expectThrow(const std::string& name, Fn&& fn)
{
    try
    {
        fn();
        report(name, false, "(未抛出异常)");
    }
    catch (const std::exception& error)
    {
        report(name, true, std::string("-> ") + error.what());
    }
}

}   // namespace

int main(int argc, char* argv[])
{
    const std::string engine_path = argc > 1 ? argv[1] : "model/model.engine";

    try
    {
        TRTInfer::InferConfig config;
        config.engine_path = engine_path;
        config.concurrency = 4;

        TRTInfer::Inference infer(config);
        std::cout << "输入元素=" << infer.inputSize()
                  << " 输出元素=" << infer.outputSize()
                  << " 并发=" << infer.concurrency() << "\n\n";

        // 构造一批互不相同的随机输入，便于检测通道间串数据
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<TRTInfer::Input> batch(16, TRTInfer::Input(infer.inputSize()));
        for (auto& input : batch)
            for (auto& value : input) value = dist(rng);

        // 1. 同步基线
        const auto baseline = infer.infer(batch[0]);

        // 2. 异步单请求
        auto async_future = infer.inferAsync(batch[0]);
        const auto async_out = async_future.get();
        report("async 与 sync 一致", same(baseline, async_out),
               "首元素=" + std::to_string(async_out.front()));

        // 3. 并行 16 请求：逐个与各自的同步结果对拍
        std::vector<TRTInfer::Output> expected;
        expected.reserve(batch.size());
        for (const auto& input : batch) expected.push_back(infer.infer(input));

        auto futures = infer.inferParallel(batch);
        report("parallel 返回数量正确", futures.size() == batch.size());

        size_t mismatched = 0;
        for (size_t i = 0; i < futures.size(); ++i)
        {
            if (!same(expected[i], futures[i].get())) ++mismatched;
        }
        report("parallel 与 sync 逐请求一致", mismatched == 0,
               "mismatch=" + std::to_string(mismatched));

        // 4. 输入元素数不匹配必须快速失败（不能起线程、不能崩）
        expectThrow("async 输入尺寸校验", [&] {
            TRTInfer::Input bad(infer.inputSize() + 1, 1.0f);
            infer.inferAsync(std::move(bad));
        });
        expectThrow("parallel 输入尺寸校验", [&] {
            std::vector<TRTInfer::Input> bad{
                TRTInfer::Input(infer.inputSize(), 1.0f),
                TRTInfer::Input(infer.inputSize() - 1, 1.0f)};
            infer.inferParallel(std::move(bad));
        });

        // 5. 在途请求数远超通道数：必须正常完成（限流而非线程爆炸/死锁）
        std::vector<std::future<TRTInfer::Output>> many;
        for (int i = 0; i < 32; ++i) many.push_back(infer.inferAsync(batch[i % batch.size()]));
        size_t many_bad = 0;
        for (size_t i = 0; i < many.size(); ++i)
        {
            if (!same(expected[i % batch.size()], many[i].get())) ++many_bad;
        }
        report("32 请求 / 4 通道 无串扰", many_bad == 0,
               "mismatch=" + std::to_string(many_bad));

        // 6. 销毁安全性：future 不 get 直接离开作用域，析构需等待在途任务
        {
            TRTInfer::Inference quick(config);
            auto dropped = quick.inferAsync(batch[0]);
            (void)dropped;   // 故意丢弃 future
        }
        report("提前销毁不崩溃", true);

        // 6b. 多线程同时调用同步 infer（借用输入，需验证不会互相踩内存）
        {
            std::vector<std::future<bool>> checks;
            for (size_t i = 0; i < batch.size(); ++i)
            {
                checks.push_back(std::async(std::launch::async, [&, i] {
                    bool ok = true;
                    for (int r = 0; r < 4; ++r) ok = ok && same(expected[i], infer.infer(batch[i]));
                    return ok;
                }));
            }
            bool all_ok = true;
            for (auto& c : checks) all_ok = c.get() && all_ok;
            report("8 线程并发同步 infer 结果正确", all_ok);
        }

        // 7. 非法配置
        expectThrow("concurrency=0 被拒绝", [&] {
            TRTInfer::InferConfig bad = config;
            bad.concurrency = 0;
            TRTInfer::Inference rejected(bad);
        });
        expectThrow("deviceId 越界被拒绝", [&] {
            TRTInfer::InferConfig bad = config;
            bad.deviceId = 9999;
            TRTInfer::Inference rejected(bad);
        });
        expectThrow("engine 路径不存在被拒绝", [&] {
            TRTInfer::InferConfig bad = config;
            bad.engine_path = "model/not-exist.engine";
            TRTInfer::Inference rejected(bad);
        });
    }
    catch (const std::exception& error)
    {
        std::cerr << "测试异常: " << error.what() << "\n";
        return 2;
    }

    std::cout << "\n" << (failures == 0 ? "全部通过" : "存在失败项") << " (failures=" << failures << ")\n";
    return failures == 0 ? 0 : 1;
}
