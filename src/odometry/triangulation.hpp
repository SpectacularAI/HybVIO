#ifndef ODOMETRY_TRIANGULATION_H_
#define ODOMETRY_TRIANGULATION_H_

#include "util.hpp"
#include "parameters.hpp"
#include "output.hpp"

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace odometry {
class EKF;
class DebugAPI;

struct CameraPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** Device position  */
    Eigen::Vector3d p;
    /** Device orientation matrices in the camera coordinate system. */
    Eigen::Matrix3d R;
    /** Derivative of R  w.r.t. each component of q */
    Eigen::Matrix3d dR[4];
    Eigen::Vector3d baseline;

    /**
     * Optional pre-triangulated 3D point information from stereo.
     *
     * The stereo pose trail two camera poses for each PIVO pose trail element:
     * first and second camera pose. If 3D features are used with stereo, this
     * flag should only be set for the first camera.
     */
    bool hasFeature3D = false;
    /** Pre-triangulated 3D point in camera coordinates, inverse depth parametrization */
    Eigen::Vector3d feature3DIdp;
    /** Pre-triangulated 3D point uncertainty in camera coordinates */
    Eigen::Matrix3d feature3DCov;
};

using CameraPoseTrail = std::vector<CameraPose, Eigen::aligned_allocator<CameraPose>>;

struct TriangulationArgsOut {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Matrix34 = Eigen::Matrix<double, 3, 4>;

    /** 3D position in world coordinates */
    Eigen::Vector3d pf;
    /** Jacobian of `pf` wrt all the pose positions */
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> dpfdp;
    /** Jacobian of `pf` wrt all the pose orientations */
    std::vector<Matrix34, Eigen::aligned_allocator<Matrix34>> dpfdq;
    /** Jacobian of `pf` wrt IMU-to-camera time shift */
    Eigen::Vector3d dpfdt;
};

struct TriangulationArgsIn {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** Normalized image points, (0, 0) means image center */
    const vecVector2d &imageFeatures;
    /** Used with `estimateImuCameraTimeShift`, in units normalized image point coordinates per second */
    const vecVector2d &featureVelocities;
    /** Camera pose trail. Same length as imageFeatures */
    const CameraPoseTrail &trail;
    /** If enabled, `imageFeatures` and `featureVelocities` contain right camera
     * inputs stacked after left inputs */
    bool stereo;
    /** Compute derivatives of the triangulation in `dpf` format */
    bool calculateDerivatives;
    /**
     * Enabling this computes imu-to-camera time shift derivative as if we had
     * shifted the features by:
     *      imageFeatures[i] += ekf->getImuToCameraTimeShift() * featureVelocities[i];
     * but in the algorithm we instead shift timestamps of camera frames in the sample
     * sync which changes the order in which samples arrive to the EKF. So the analytical
     * and numeric derivatives of the triangulation will not match when this option is enabled.
     */
    bool estimateImuCameraTimeShift;
    /** Enable in derivative-checking unit tests to make the function match the derivatives */
    bool derivativeTest = false;
    /** Used for computing derivatives when `derivativeTest` is true */
    double imuToCameraTimeShift = 0.0;
};

struct TwoCameraTriangulationArgsIn {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** First pose to use. */
    const CameraPose &pose0;
    /** Second pose to use. */
    const CameraPose &pose1;
    /** `imageFeatures` of first pose. */
    const Eigen::Vector2d &ip0;
    /** `imageFeatures` of second pose. */
    const Eigen::Vector2d &ip1;
    /** `featureVelocities` of first pose. */
    const Eigen::Vector2d *velocity0 = nullptr;
    /** `featureVelocities` of second pose. */
    const Eigen::Vector2d *velocity1 = nullptr;
    // The rest are as in `TriangulationArgsIn`.
    bool calculateDerivatives = false;
    bool estimateImuCameraTimeShift = false;
    bool derivativeTest = false;
    double imuToCameraTimeShift = 0.0;
};

struct PrepareVisualUpdateArgsIn {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const TriangulationArgsOut &triangulationOut;
    const vecVector2d &featureVelocities;
    const CameraPoseTrail &trail;
    const std::vector<int> &poseTrailIndex;
    int stateDim;
    bool useStereo;
    bool truncated;
    int mapPointOffset;
    // The rest are as in `TriangulationArgsIn`.
    bool estimateImuCameraTimeShift = false;
    bool derivativeTest = false;
    double imuToCameraTimeShift = 0.0;
};

/**
 * Extracts the poses whose indices are given in poseTrailIndex from the
 * odometry state vector. In poseTrailIndex, 0 means the current pose and
 * the last possible pose has index camTrailLength+1. Note that these are
 * not raw indices of the odometry state vector.
 *
 * 3D feature information is left unset by this function.
 */
void extractCameraPoseTrail(
    const odometry::EKF& ekf,
    const std::vector<int> &poseTrailIndex,
    const Parameters &parameters,
    bool useStereo,
    CameraPoseTrail& trail);

class Triangulator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Matrix3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
    using Matrix32 = Eigen::Matrix<double, 3, 2>;
    using Matrix23 = Eigen::Matrix<double, 2, 3>;

    static constexpr unsigned POSE_DIM = 7;

    /**
     * An iterative triangulation algorithm.
     *
     * @param args common triangulation input arguments
     * @param out triangulation output
     * @param odometryDebugAPI
     * @return triangulation status, OK if successful
     */
    TriangulatorStatus triangulate(
        const TriangulationArgsIn &args,
        TriangulationArgsOut &out,
        odometry::DebugAPI *odometryDebugAPI = nullptr);

    Triangulator(const ParametersOdometry &parameters);

private:
    TriangulatorStatus triangulateStereo(
        const TriangulationArgsIn &args,
        TriangulationArgsOut &out,
        odometry::DebugAPI *odometryDebugAPI = nullptr);

    const ParametersOdometry &parameters;
    using MatrixX3 = Eigen::Matrix<double, Eigen::Dynamic, 3>;
    // Temporaries for iterative triangulation.
    Eigen::MatrixXd dEerror, dETE;
    Matrix3X dpfi;
    // for triangulateStereo
    Matrix3X dWeightedSum;
    Eigen::MatrixXd dSumOfWeights;
};

/**
 * Compute a rough triangulation estimate with just two poses.
 *
 * @param dpf Derivatives of return value `pf` in order (p[0], q[0], p[1], q[1], t).
 * @return The triangulated 3d point, given in the the camera coordinates (p[0], R[0])
 *         and always lies on the ray defined by imageFeature0.
 */
Eigen::Vector3d triangulateWithTwoCameras(
    const TwoCameraTriangulationArgsIn &args,
    Eigen::Matrix<double, 3, 2 * Triangulator::POSE_DIM + 1> *dpf = nullptr);

/**
 * Triangulate stereo feature in first camera coordinates
 *
 * @param normalizedPixelFirst normalized pixel coords of the first camera ray
 * @param normalizedPixelSecond second feature
 * @param secondToFirstCamera 4x4 homogeneous matrix that translates a point
 *  from the coordinate system of the second camera to that of the first camera
 * @param triangulatedPoint (output): coordinates of the result in the first
 *  camera coordinate system, inverse depth parametrization [x/z, y/z, 1/z]
 * @param triangulatedCov (output): if not `nullptr`, uncertainty of the triangulation result
 * @return true on success. Then the outputs are also set. Otherwise they are
 *  undefined.
 */
bool triangulateStereoFeatureIdp(
    const Eigen::Vector2d &normalizedPixelFirst,
    const Eigen::Vector2d &normalizedPixelSecond,
    const Eigen::Matrix4d &secondToFirstCamera,
    Eigen::Vector3d &triangulatedPoint,
    Eigen::Matrix3d *triangulatedCov);

/**
 * Algorithm from the book Computer Vision: Algorithms and Applications
 * by Richard Szeliski. Chapter 7.1 Triangulation, page 345.
 *
 * @param args common triangulation input arguments
 * @param out triangulation output
 * @return triangulation status, OK if successful
 */
TriangulatorStatus triangulateLinear(
    const TriangulationArgsIn &args,
    TriangulationArgsOut &out);

PrepareVuStatus prepareVisualUpdate(
    const PrepareVisualUpdateArgsIn &args,
    Eigen::MatrixXd &H,
    Eigen::VectorXd &y);

void getPosOriIndices(int i, int& pos, int& ori);

Triangulator::Matrix23 pinv(const Triangulator::Matrix32& A);
/**
 * Convert a point from camera coordinates to inverse depth representation
 * or back (note that this function is its own inverse).
 * @param p 3D point in camera coordinates
 * @param dip Jacobian of the p -> ip mapping
 * @param ddip if non-null, must be point to the head of a 3-element array like
 *      Eigen::Matrix3d ddip[3], whose elements will be set to the second
 *      derivative matrices of the transform
 * @return ip, the inverse depth parametrized point
 */
Eigen::Vector3d inverseDepth(const Eigen::Vector3d &p, Eigen::Matrix3d &dip, Eigen::Matrix3d *ddip = nullptr);

} // namespace odometry

#endif // ODOMETRY_TRIANGULATION_H_
