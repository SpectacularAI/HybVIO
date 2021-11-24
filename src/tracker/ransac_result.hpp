#ifndef TRACKER_RANSAC_RESULT_H_
#define TRACKER_RANSAC_RESULT_H_

#include <vector>
#include "track.hpp"
#include <opencv2/core.hpp>

namespace tracker {
struct RansacResult {
    enum class Type : uint8_t {
        SKIPPED = 0,
        R2 = 1,
        R3 = 2,
        R5 = 3,
        UPRIGHT_2P = 4
    };

    Type type = Type::SKIPPED;
    size_t inlierCount = 0;

    // these are undefined if type = Skipped
    cv::Matx33d R; // rotation matrix
    cv::Matx31d t; // translation

    // Not used for RANSAC3.
    std::vector<Feature::Status> inliers;

    void initialize(size_t nTrackedFeatures);

    // TODO: this is quite tightly coupled with TrackerImplementation
    void updateTrackStatus(std::vector<Feature::Status> &trackStatus) const;
};
}

#endif
