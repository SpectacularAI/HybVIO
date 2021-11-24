#ifndef DAZZLING_PROCESSED_FRAME_H_
#define DAZZLING_PROCESSED_FRAME_H_

#include <memory>
#include <Eigen/Dense>

#include "sample_sync.hpp"

namespace odometry {
struct ProcessedFrame {
    ~ProcessedFrame();
    ProcessedFrame(
        double t,
        ImagePtr firstGrayFrame,
        ImagePtr secondGrayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    );

    // Timestamp.
    double t;
    // Which leader sensor sample the frame should be paired with.
    size_t leaderIndex;
    // Current best time difference to a leader sample.
    double leaderTimeDiff;
    // Running numbering for frames inserted into SampleSync.
    size_t num;
    // Grayscale camera frames.
    std::shared_ptr<tracker::Image> firstGrayFrame, secondGrayFrame;
    // Optional data. Can be `nullptr`.
    std::unique_ptr<TaggedFrame> taggedFrame;
};
}

#endif
