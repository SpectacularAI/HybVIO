#ifndef ODOMETRY_DEBUG_HPP
#define ODOMETRY_DEBUG_HPP

#include <Eigen/StdVector>
#include <Eigen/Dense>

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "util.hpp"
#include "../api/vio.hpp"

namespace slam {
struct DebugAPI;
class MapPointRecord;
}

namespace odometry {
class EKF;
class EKFStateIndex;
struct Parameters;

struct DebugPublisher {
    virtual void startFrame(
        const EKF &ekf,
        const EKFStateIndex &ekfStateIndex,
        const Parameters &parameters
    ) = 0;
    virtual void startVisualUpdate(
        double age,
        const EKF &ekf,
        const std::vector<int> &poseTrailIndex,
        const vecVector2d &imageFeatures,
        const Parameters &parameters
    ) = 0;
    virtual void pushTriangulationPoint(const Eigen::Vector3f &p) = 0;
    virtual void finishSuccessfulVisualUpdate(
        const EKF &ekf,
        const std::vector<int> &poseTrailIndex,
        const vecVector2d &imageFeatures,
        const Parameters &parameters
    ) = 0;
    virtual void addSample(double t, const Eigen::Vector3f &g, const Eigen::Vector3f &a) = 0;
    virtual void addPointCloud(const vecVector3f &pointsCamCoords, const vecVector3f *pointColors = nullptr) = 0;
};

class DebugAPI {
public:
    slam::DebugAPI* getSlamDebugApi() const {
#ifdef USE_SLAM
        if (slamDebug) {
            return &*slamDebug;
        }
#endif
        return nullptr;
    }

    DebugPublisher *publisher = nullptr;
    std::function<void(const std::map<int, slam::MapPointRecord>&)> endDebugCallback = nullptr;
#ifdef USE_SLAM
    std::unique_ptr<slam::DebugAPI> slamDebug;
#else
    slam::DebugAPI *slamDebug = nullptr;
#endif
};

} // namespace odometry

#endif
