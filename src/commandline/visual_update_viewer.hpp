#ifndef DAZZLING_VISUAL_UPDATE_VIEWER_HPP
#define DAZZLING_VISUAL_UPDATE_VIEWER_HPP

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <map>
#include "../api/internal.hpp"

class CommandQueue;
namespace cmd { struct Parameters; }

namespace odometry {
struct DebugPublisher;

namespace viewer {

struct VisualUpdateViewer {
    using PoseHistory = api::PoseHistory;
    using PoseHistoryMap = std::map<PoseHistory, std::vector<api::Pose>>;

    static std::unique_ptr<VisualUpdateViewer> create(const cmd::Parameters &parameters, CommandQueue &commands);
    virtual ~VisualUpdateViewer() = 0;

    // Run on rendering thread.
    virtual void setup() = 0;
    virtual void setupFixedData() = 0;
    virtual void draw() = 0;

    // Run on algorithm thread.
    virtual void setFixedData(
        const PoseHistoryMap &poseHistories,
        const Eigen::Matrix4d &imuToCamera,
        const Eigen::Matrix4d &secondImuToCamera
    ) = 0;

    virtual DebugPublisher& getPublisher() = 0;
};

} // namespace viewer
} // namespace odometry

#endif // DAZZLING_VISUAL_UPDATE_VIEWER_HPP
