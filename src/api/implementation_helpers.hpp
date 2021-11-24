#ifndef DAZZLING_API_IMPLEMENTATION_HELPERS_HPP
#define DAZZLING_API_IMPLEMENTATION_HELPERS_HPP

#include "internal.hpp"
#include "../tracker/util.hpp"
#include "../odometry/util.hpp"

namespace api {
namespace { // internal helper classes and functions
// TODO: Remove
InternalAPI::DebugParameters applyAutoParameters(const InternalAPI::DebugParameters &original) {
    InternalAPI::DebugParameters p(original);
    tracker::util::automaticCameraParametersWhereUnset(p.api.parameters);
    tracker::util::scaleImageParameters(p.api.parameters.tracker, p.videoAlgorithmScale);

    // odometry visual updates require this: tracks can't be longer. On the other hand,
    // there is no point making them shorter as then the end of the camera pose trail is useless
    p.api.parameters.tracker.maxTrackLength = p.api.parameters.odometry.cameraTrailLength + 1;

    return p;
}

void convertViotesterAndroidPose(Pose &pose, const Eigen::Matrix4d &imuToCamera) {
    // TODO ... review all of this. Looks too complex!

    // This is used in viotester for rendering.
    Eigen::Matrix4d androidTransform; androidTransform <<
        1,  0,  0,  0,
        0,  0,  1,  0,
        0, -1,  0,  0,
        0,  0,  0,  1;

    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Vector3d p(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    // The result was looked correct expect it pointed the opposite direction so I
    // added this.
    Eigen::Matrix3d U; U <<
        1, 0, 0,
        0, -1, 0,
        0, 0, -1;
    R = U * R;
    // Reverse transformation viotester performs before recording.
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = -R * p;
    T.block<3, 3>(0, 0) = R;
    // Transform to our world-to-camera coordinates.
    T = T * androidTransform;
    // Transform to our IMU coordinates.
    Eigen::Vector4d orientation = odometry::util::rmat2quat(imuToCamera.topLeftCorner<3, 3>().inverse() * T.topLeftCorner<3, 3>());
    Eigen::Vector3d position = -T.topLeftCorner<3, 3>().transpose() * T.block<3, 1>(0, 3);

    pose.orientation.w = orientation(0);
    pose.orientation.x = orientation(1);
    pose.orientation.y = orientation(2);
    pose.orientation.z = orientation(3);
    pose.position.x = position(0);
    pose.position.y = position(1);
    pose.position.z = position(2);
}

double setIntrinsic(
    const std::string &kind,
    double perFrame,
    double custom,
    double automatic = -1.0
) {
    if (custom > 0.0) {
        return custom;
    }
    else if (perFrame > 0.0) {
        return perFrame;
    }
    else if (automatic > 0.0) {
        return automatic;
    }
    log_error("setIntrinsic(): No intrinsic value available: %s.", kind.c_str());
    assert(false);
    return -1.0;
}

}
}

#endif
