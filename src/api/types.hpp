#ifndef DAZZLING_TYPES_HPP
#define DAZZLING_TYPES_HPP

// Common types, can be included "everywhere"
#include "jsonl-recorder/types.hpp"

namespace api {
using Vector3d = recorder::Vector3d;
using Quaternion = recorder::Quaternion;
using Pose = recorder::Pose;

/**
 * A 3x3 matrix, row major (accessed as m[row][col]).
 * Also note that when the matrix is symmetric (like covariance matrices),
 * there is no difference between row-major and column-major orderings.
 */
using Matrix3d = std::array<std::array<double, 3>, 3>;

/** An element of the point cloud */
struct FeaturePoint {
    /**
     * An integer ID to identify same points in different
     * revisions of the point clouds
     */
    int id;

    /** Global position of the feature point */
    Vector3d position;

    /** Implementation-defined status/type */
    int status = 0;
};

enum class TrackingStatus {
    INIT = 0, // Initial status when tracking starts and is still initializing
    TRACKING = 1, // When tracking is accurate
    LOST_TRACKING = 2 // When tracking fails after having achieved TRACKING state
};

struct CameraParameters {
    // Negative values denote uninitialized.
    double focalLengthX = -1.0;
    double focalLengthY = -1.0;
    double principalPointX = -1.0;
    double principalPointY = -1.0;

    CameraParameters() {}
    CameraParameters(double focalLength) {
        focalLengthX = focalLength;
        focalLengthY = focalLength;
    }

    // TODO: distortion coefs etc.
};
}

#endif
