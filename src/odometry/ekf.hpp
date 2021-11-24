#ifndef ODOMETRY_EKF_H_
#define ODOMETRY_EKF_H_

#include <Eigen/Dense>
#include <memory>

namespace odometry {

// Structure of the EKF state variable.
//
// pose = [
// 0-2:   p0, p1, p2,
// 3-6:   qw, qx, qy, qz,
//        ]
//
// state m = [
// 0-2:   p0, p1, p2,
// 3-5:   v0, v1, v2,
// 6-9:   qw, qx, qy, qz,
// 10-12: bga0, bga1, bga2,
// 13-15: baa0, baa1, baa2,
// 16-18: bat0, bat1, bat2,
// 19:    imuToCameraTimeShift,
// 20->:  {poses}, {map points}
//        ]
constexpr int POS = 0;
constexpr int VEL = 3;
constexpr int ORI = 6;
constexpr int BGA = 10;
constexpr int BAA = 13;
constexpr int BAT = 16;
constexpr int SFT = 19;
constexpr int CAM = 20;
constexpr int INER_DIM = CAM;
constexpr int POSE_DIM = 7;
constexpr int MAP_POINT_DIM = 3;
// BGA = Bias Gyro Additive.
// BAA = Bias Acc Additive.
// BAT = Bias Acc Transform.
constexpr int INER_VAR_COUNT = 7;
constexpr std::array<int, INER_VAR_COUNT> STATE_PARTS = {POS, VEL, ORI, BGA, BAA, BAT, SFT};
const std::array<std::string, INER_VAR_COUNT> STATE_PART_NAMES = {"POS", "VEL", "ORI", "BGA", "BAA", "BAT", "SFT"};
constexpr std::array<int, INER_VAR_COUNT> STATE_PART_SIZES = {3, 3, 4, 3, 3, 3, 1};

// Structure of process noise (prediction).
constexpr int Q_ACC = 0;
constexpr int Q_GYRO = 3;
constexpr int Q_BGA_DRIFT = 6;
constexpr int Q_BAA_DRIFT = 9;
constexpr int Q_DIM = 12;

struct Parameters; // forward declaration

enum class VuOutlierStatus {
    INLIER,
    NOT_COMPUTED,
    RMSE,
    CHI2,
};

// Extended Kalman Filter (EKF) implementing the odometry.
class EKF {
public:
    static std::unique_ptr<EKF> build(const Parameters &parameters);
    virtual std::unique_ptr<EKF> clone() const = 0; // for tests
    virtual ~EKF();

    virtual void initializeOrientation(const Eigen::Vector3d &xa) = 0;
    virtual void predict(double t, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa) = 0;
    virtual Eigen::Vector3d position() const = 0;
    virtual Eigen::Vector3d velocity() const = 0;
    virtual Eigen::Vector4d orientation() const = 0;
    virtual Eigen::Vector3d biasGyroscopeAdditive() const = 0;
    virtual Eigen::Vector3d biasAccelerometerAdditive() const = 0;
    virtual Eigen::Vector3d biasAccelerometerTransform() const = 0;
    virtual int camTrailSize() const = 0;
    virtual Eigen::Vector3d historyPosition(int i) const = 0;
    virtual Eigen::Vector4d historyOrientation(int i) const = 0;
    virtual double historyTime(int i) const = 0;
    virtual double speed() const = 0;
    virtual double horizontalSpeed() const = 0;
    virtual void updateZupt(double r) = 0;
    virtual void updateZuptInitialization() = 0;
    virtual void updateZrupt(const Eigen::Vector3d &xg) = 0;
    virtual void updatePseudoVelocity(double defaultSpeed, double r) = 0;
    virtual void updatePosition(const Eigen::Vector3d &pos, double r) = 0;
    virtual void updateZeroHeight(double r) = 0;
    virtual void updateOrientation(const Eigen::Vector4d &q, double r) = 0;

    using MatrixInertialCov = Eigen::Matrix<double, INER_DIM, INER_DIM>;
    using VectorInertialMean = Eigen::Matrix<double, INER_DIM, 1>;

    virtual void getInertialState(VectorInertialMean &mean, MatrixInertialCov &cov) const = 0;
    virtual void setInertialState(const VectorInertialMean &mean, const MatrixInertialCov &cov) = 0;

    virtual double getImuToCameraTimeShift() const = 0;

    /**
     * Translate the current and each historical pose by deltaX so that the
     * current position matches with the given pos. Does not alter the other
     * elements in the state or the covariance.
     */
    virtual void translateTo(const Eigen::Vector3d &pos) = 0;

    /**
     * Rotate and translate all elements in the pose trail, and velocity,
     * so that the pose with index i becomes equal to the given pose. This
     * is a heavier operation than translate_to. The default pose index of
     * -1 means using the latest pose.
     *
     * This is as close as you can get to rotating / changing the coordinate
     * system without affecting anything else. However, note that applying
     * rotations that tilt the z-axis do affect the subsequent behavior of the
     * algorithm because the accelerometer data gives information about the
     * absolute z-axis direction due to gravity.
     */
    virtual void transformTo(const Eigen::Vector3d &pos, const Eigen::Vector4d &q, int i = -1) = 0;

    virtual VuOutlierStatus visualTrackOutlierCheck(
        const Eigen::MatrixXd &visH,
        const Eigen::VectorXd &f,
        const Eigen::VectorXd &y,
        double r,
        double trackRmseThreshold) = 0;

    virtual void updateVisualTrack(
        const Eigen::MatrixXd &visH,
        const Eigen::VectorXd &f,
        const Eigen::VectorXd &y,
        double r) = 0;

    /**
     * Clone the current pose, move the K-1 subsequent poses one slot forward
     * in the state, drop pose K and leave the remaining poses unchanged
     * @discardedPoseIndex index of the dropped pose, K, where 0 means the
     *      first historical pose. -1 means the latest pose in the trail.
     */
    virtual void updateVisualPoseAugmentation(int discardedPoseIndex = -1) = 0;
    virtual void updateUndoAugmentation() = 0;

    // Hybrid EKF

    // get map point at given slot, as well as its derivative w.r.t. all state variables
    virtual Eigen::Vector3d getMapPoint(int idx) const = 0;
    virtual void insertMapPoint(int idx, const Eigen::Vector3d &pf) = 0;
    virtual int getMapPointStateIndex(int idx) const = 0;

    // Internals
    virtual void conditionOnLastPose() = 0;
    virtual void lockBiases() = 0;
    virtual void normalizeQuaternions(bool onlyCurrent) = 0;
    virtual void setFirstSampleTime(double t) = 0;
    virtual bool isPositiveSemiDefinite() = 0;
    virtual void maintainPositiveSemiDefinite() = 0;
    virtual void setState(const Eigen::VectorXd &m) = 0;
    virtual void setStateCovariance(const Eigen::MatrixXd &P) = 0;
    virtual void setProcessNoise(const Eigen::MatrixXd &Q) = 0;
    virtual double getPlatformTime() const = 0;
    virtual int getPoseCount() const = 0;
    virtual const Eigen::VectorXd &getState() const = 0;
    virtual Eigen::MatrixXd getStateCovariance() const = 0;
    virtual const Eigen::MatrixXd& getStateCovarianceRef() const = 0;
    virtual Eigen::MatrixXd getVisAugH() const = 0;
    virtual Eigen::MatrixXd getVisAugA() const = 0;
    virtual Eigen::MatrixXd getVisAugQ() const = 0;
    virtual Eigen::MatrixXd getDydx() const = 0;
    virtual std::string stateAsString() const = 0;
    virtual int getStateDim() const = 0;
    virtual bool getWasStationary() const = 0;

protected:
    EKF(const EKF &other); // for clone / tests
    EKF();
};
} // namespace odometry

#endif // ODOMETRY_EKF_H_
