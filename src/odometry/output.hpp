#ifndef ODOMETRY_OUTPUT_HPP_
#define ODOMETRY_OUTPUT_HPP_

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <memory>
#include "../api/types.hpp" // for trackingStatus... questionable

// Output types for odometry, some are internal
namespace odometry {
struct TaggedFrame;
class EKF;
class EKFStateIndex;

enum PrepareVuStatus {
    PREPARE_VU_OK = 0,
    PREPARE_VU_ZERO_DEPTH = 1,
    PREPARE_VU_BEHIND = 2
};

enum class TriangulatorStatus : uint8_t {
    OK = 0,
    HYBRID, // i.e., skipped
    BEHIND,
    BAD_COND,
    NO_CONVERGENCE,
    BAD_DEPTH,
    UNKNOWN_PROBLEM
};

struct PointFeature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id;
    enum class Status {
        UNUSED = 0,
        POSE_TRAIL = 1,
        HYBRID = 2,
        SLAM = 3,
        OUTLIER = 4,
        STEREO = 5
    } status;
    Eigen::Vector2f firstPixel; // (-1, -1) if not available
    Eigen::Vector3d point;
};

class Output {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Output();

    double t;
    float focalLength;
    bool stationaryVisual;
    api::TrackingStatus trackingStatus;

    Eigen::Vector3d position() const;
    Eigen::Vector3d velocity() const;
    Eigen::Vector4d orientation() const;
    Eigen::Matrix3d positionCovariance() const;
    Eigen::Matrix3d velocityCovariance() const;
    Eigen::Vector3d meanBGA() const;
    Eigen::Vector3d meanBAA() const;
    Eigen::Vector3d meanBAT() const;
    Eigen::Vector3d covDiagBGA() const;
    Eigen::Vector3d covDiagBAA() const;
    Eigen::Vector3d covDiagBAT() const;

    // If pose trail is not available / set in this object, returns 0
    // This version of the pose trail only includes the "historical poses"
    size_t poseTrailLength() const;
    Eigen::Vector3d poseTrailPosition(int idx) const;
    Eigen::Vector4d poseTrailOrientation(int idx) const;
    double poseTrailTimeStamp(int idx) const;
    size_t poseTrailOffset(int idx) const;

    using PointCloud = std::vector< PointFeature, Eigen::aligned_allocator<PointFeature> >;
    std::shared_ptr<PointCloud> pointCloud;
    std::shared_ptr<TaggedFrame> taggedFrame;

    void setFromEKF(const EKF &ekf, const EKFStateIndex &stateIndex,
        std::shared_ptr<Eigen::VectorXd> fullMeanStore = {},
        std::shared_ptr<std::vector<double>> poseTrailTimeStampsStore = {});

    // also very ugly and only necessary due to odometry<->SLAM coordinate transforms
    void addPoseTrailElementMeanOnly(
        int idx,
        double t,
        const Eigen::Vector3d &pos,
        const Eigen::Vector4d &ori);

private:
    // A low-dimensional subset of the state that is always stored
    Eigen::Matrix3d positionCov;
    Eigen::Matrix3d velocityCov;
    static constexpr int INER_DIM = 20;
    Eigen::Matrix<double, INER_DIM, 1> inertialMean;
    Eigen::Matrix<double, INER_DIM, 1> inertialCovDiag;

    // These may be empty
    std::shared_ptr<Eigen::VectorXd> fullMean;
    std::shared_ptr<std::vector<double>> poseTrailTimeStamps;
};
}

#endif
