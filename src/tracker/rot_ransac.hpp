#ifndef TRACKER_ROT_RANSAC_H_
#define TRACKER_ROT_RANSAC_H_

#include "track.hpp"
#include <random>
#include <memory>
#include <array>
#include <opencv2/core.hpp>

namespace tracker {
class Camera;
namespace rot_ransac {

class RotRansac {
public:
    RotRansac();
    ~RotRansac();

    cv::Matx33f fit(
        const std::vector<Feature::Point> &c1,
        const std::vector<Feature::Point> &c2,
        const Camera &camera1,
        const Camera &camera2,
        std::vector<Feature::Status> &bestInliers,
        std::mt19937& rng);

    bool withinInlierThreshold(Feature::Point a, Feature::Point b) const;

    size_t bestInlierCount;
    // Squared distance in pixels that a rotated point must be within the corresponding
    // point to be considered an inlier.
    float threshold_pow2;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

cv::Matx33f solveRotation(const std::vector<cv::Matx31f>& p1, const std::vector<cv::Matx31f>& p2, const std::vector<std::size_t>& inds);

} // namespace rot_ransac
} // namespace tracker

#endif // TRACKER_ROT_RANSAC_H_
