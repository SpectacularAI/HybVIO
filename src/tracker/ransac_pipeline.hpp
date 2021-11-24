#ifndef TRACKER_RANSAC_PIPELINE_H_
#define TRACKER_RANSAC_PIPELINE_H_

#include "track.hpp"

#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace odometry { struct Parameters; }

namespace tracker {
struct RansacResult;
class Camera;

class RansacPipeline {
public:
    static std::unique_ptr<RansacPipeline> build(
        int imageWidth, int imageHeight,
        const odometry::Parameters &parameters);

    virtual ~RansacPipeline() = default;

    /**
     * Filter tracks with RANSAC methods
     *
     * @param cameras vector length of number of cameras of previous and current camera models.
     * @param corners vector length of number of cameras of previous and current 2D points.
     *  All the inner vectors have same size N.
     * @param poses previous and current camera pose estimates. Can be `nullptr`.
     * @param trackStatus status vector, must have the same size N.
     *  only points with the status FS_TRACKED at the corresponding location in
     *  this vector are considered. This is also the output vector, points with
     *  status FS_TRACKED that fail the RANSAC tests are marked with status
     *  FS_RANSAC_OUTLIER.
     * @return a "stationarity score" [0,1], for example, RANSAC-2 inlier ratio
     */
    virtual double compute(
        const std::vector<std::array<const Camera*, 2>> &cameras,
        const std::vector<std::array<const std::vector<Feature::Point>*, 2>> &corners,
        const std::array<Eigen::Matrix4d, 2> *poses,
        std::vector<Feature::Status> &trackStatus) = 0;

    // for visualization purposes
    virtual const RansacResult &lastResult() const = 0;
};
}

#endif
