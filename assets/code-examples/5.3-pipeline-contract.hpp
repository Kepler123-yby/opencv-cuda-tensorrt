#pragma once
// Interface scaffold only. No CPU/CUDA/TensorRT backend implementation here.
#include <opencv2/core.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace coach {
struct Frame {
    std::int64_t id;
    cv::Mat bgr; // CV_8UC3, owning/reference-counted storage; immutable while in flight
    std::int64_t ready_ns;
};
struct Detection {
    float x1, y1, x2, y2, score; // source-image coordinates
    int class_id;
    int candidate_id; // deterministic tie-break and parity alignment
};
struct RequestToken {
    std::size_t slot;
    std::uint64_t generation; // detects stale handles after slot reuse
    std::int64_t frame_id;
};
struct Result {
    std::int64_t frame_id;
    cv::Mat original; // same immutable frame as input; clone before drawing
    std::vector<Detection> detections;
    std::int64_t ready_ns, submit_ns, done_ns;
};
class Pipeline {
public:
    virtual ~Pipeline() = default;
    virtual std::size_t capacity() const noexcept = 0;
    // Needs a free slot. Must throw on exhaustion; never overwrite a live request.
    // Implementation stores the frame/geometry until collect finishes.
    virtual RequestToken submit(Frame frame) = 0;
    // Query without consuming. Validate slot/generation/frame_id; propagate errors.
    virtual bool ready(const RequestToken& token) = 0;
    // Block for this request, transfer candidates, run CPU NMS and timestamp done.
    // Consume exactly once; release the slot only after host results are owned.
    virtual Result collect(const RequestToken& token) = 0;
};
} // namespace coach
