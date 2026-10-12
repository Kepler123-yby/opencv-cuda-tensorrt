// Schema/relationship validation only: does NOT probe GPU, load ONNX/engine,
// or certify DNN operator support. Build: g++ -std=c++20 ... -lyaml-cpp
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void keys(const YAML::Node& node, const std::set<std::string>& allowed,
          const std::string& scope) {
    require(node.IsMap(), scope + " must be a mapping");
    std::set<std::string> seen;
    for (const auto& entry : node) {
        std::string key = entry.first.as<std::string>();
        require(allowed.count(key) != 0, scope + ": unknown key " + key);
        require(seen.insert(key).second, scope + ": duplicate key " + key);
    }
    for (const auto& key : allowed)
        require(bool(node[key]), scope + ": missing key " + key);
}
void oneOf(const std::string& value, const std::set<std::string>& choices,
           const std::string& name) {
    require(choices.count(value) != 0, name + ": invalid value " + value);
}
std::string text(const YAML::Node& node, const std::string& key) {
    const auto field = node[key];
    require(field.IsScalar() && !field.IsNull(), "expected non-null scalar field: " + key);
    std::string value = field.as<std::string>();
    require(!value.empty(), "empty field: " + key);
    return value;
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: validate-config PATH.yaml");
        fs::path config_path = fs::absolute(argv[1]).lexically_normal();
        YAML::Node cfg = YAML::LoadFile(config_path.string());
        keys(cfg, {"schema_version", "model", "runtime", "preprocess", "postprocess", "benchmark"}, "root");
        require(cfg["schema_version"].as<int>() == 1, "unsupported schema_version");
        auto model = cfg["model"], rt = cfg["runtime"], pre = cfg["preprocess"];
        auto post = cfg["postprocess"], bench = cfg["benchmark"];
        keys(model, {"onnx", "engine", "input_name", "output_name", "input_shape"}, "model");
        keys(rt, {"backend", "cuda_enabled", "device_id", "precision", "mode", "slots"}, "runtime");
        keys(pre, {"implementation", "pad_value"}, "preprocess");
        keys(post, {"implementation", "output_layout", "num_classes", "score_threshold", "nms_iou"}, "postprocess");
        keys(bench, {"warmup", "frames"}, "benchmark");

        const auto shape = model["input_shape"].as<std::vector<int>>();
        require(shape.size() == 4, "input_shape must be [N,C,H,W]");
        require(shape[0] == 1 && shape[1] == 3 && shape[2] > 0 && shape[3] > 0,
                "this example requires positive [1,3,H,W]");
        text(model, "input_name"); text(model, "output_name");
        auto pathFromYaml = [&](const std::string& key) {
            fs::path path(text(model, key));
            return (path.is_absolute() ? path : config_path.parent_path() / path).lexically_normal();
        };
        const fs::path onnx = pathFromYaml("onnx"), engine = pathFromYaml("engine");
        const std::string backend = text(rt, "backend"), precision = text(rt, "precision");
        const std::string mode = text(rt, "mode"), pre_impl = text(pre, "implementation");
        const std::string post_impl = text(post, "implementation");
        const bool cuda = rt["cuda_enabled"].as<bool>();
        const int slots = rt["slots"].as<int>();
        oneOf(backend, {"opencv_cpu", "opencv_cuda", "tensorrt"}, "backend");
        oneOf(precision, {"fp32", "fp16"}, "precision");
        oneOf(mode, {"sync", "async", "parallel"}, "mode");
        oneOf(pre_impl, {"cpu", "opencv_cuda", "fused_cuda"}, "preprocess");
        oneOf(post_impl, {"cpu", "cuda_decode"}, "postprocess");
        oneOf(text(post, "output_layout"), {"candidate_major", "feature_major"}, "output_layout");
        require(rt["device_id"].as<int>() >= 0, "device_id must be non-negative");
        require(slots >= 1 && slots <= 8, "slots must be 1..8 (teaching limit)");
        require(mode == "parallel" ? slots >= 2 : slots == 1, "mode/slots mismatch");
        require(backend == "tensorrt" || mode == "sync", "async/parallel supported here only for TensorRT");
        require(backend == "tensorrt" || precision == "fp32", "this OpenCV example supports fp32 only");
        const bool needs_gpu = backend != "opencv_cpu" || pre_impl != "cpu" || post_impl != "cpu";
        require(!needs_gpu || cuda, "GPU path requested while cuda_enabled=false");
        require(pre_impl != "fused_cuda" || backend == "tensorrt", "fused_cuda here writes TensorRT device input");
        require(post_impl != "cuda_decode" || backend == "tensorrt", "cuda_decode here consumes TensorRT device output");
        const double pad = pre["pad_value"].as<double>();
        require(std::isfinite(pad) && pad >= 0 && pad <= 255 && std::floor(pad) == pad,
                "pad_value must be an integer 0..255 for the uint8 reference");
        require(post["num_classes"].as<int>() > 0, "num_classes must be positive");
        for (const char* name : {"score_threshold", "nms_iou"}) {
            double value = post[name].as<double>();
            require(std::isfinite(value) && value >= 0 && value <= 1, std::string(name) + " must be finite in [0,1]");
        }
        require(bench["warmup"].as<int>() >= 0, "warmup must be non-negative");
        require(bench["frames"].as<int>() > 0, "frames must be positive");
        std::cout << "PASS: schema and configuration relationships\n"
                  << "backend=" << backend << " cuda_enabled=" << (cuda ? "true" : "false")
                  << " requires_gpu=" << (needs_gpu ? "true" : "false") << "\n"
                  << "preprocess=" << pre_impl << " postprocess=" << post_impl
                  << " mode=" << mode << " slots=" << slots << "\n"
                  << "onnx=" << onnx.string() << "\nengine=" << engine.string() << "\n"
                  << "NOT CHECKED: files, compiled capabilities, device, model IO, inference\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CONFIG ERROR: " << e.what() << "\n";
        return 2;
    }
}
