#ifndef TRACKER_TRACK_H_
#define TRACKER_TRACK_H_

#include <array>

namespace tracker {

struct Feature {
    enum class Status {
        TRACKED,
        NEW,
        // The rest are all failure cases.
        FAILED_FLOW,
        RANSAC_OUTLIER,
        FLOW_OUT_OF_RANGE,
        OUT_OF_RANGE,
        FAILED_EPIPOLAR_CHECK,
        CULLED,
        BLACKLISTED
    };

    struct Point {
        float x, y;
    };

    /** Unique track ID, must be set */
    int id = -1;
    Status status = Status::NEW;
    // Coordinates for left and right frames.
    std::array<Point, 2> points = {{ { -1, -1 }, { -1, -1 } }};
    float depth = -1;
};

} // namespace tracker

#endif // TRACKER_TRACK_H_
