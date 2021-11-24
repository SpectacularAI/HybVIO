// #define EIGEN_RUNTIME_NO_MALLOC

#include "ekf.hpp"

#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/MatrixFunctions>

#include "parameters.hpp"
#include "util.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"

namespace {
using namespace odometry;

// --- Static helper methods

// Second half of a general KF update.
void updateCommon(Eigen::VectorXd& m, Eigen::MatrixXd& P, const Eigen::MatrixXd& HP, const Eigen::MatrixXd& K)
{
    P.noalias() -= K * HP;

    // TODO: it's a hack to do this here
    // Normalize orientation quaternion. Camera trail quaternions are not normalized here.
    m.segment(ORI, 4).normalize();
}

// Second half of a general KF update (alternative version)
void updateCommonJosephForm(Eigen::MatrixXd& P,
                   const Eigen::MatrixXd& H, const Eigen::MatrixXd& R, const Eigen::MatrixXd& K,
                   Eigen::MatrixXd& tmpP1, Eigen::MatrixXd& tmpP0) {
    const int n = P.rows();
    assert(H.cols() == n);

    tmpP1.noalias() = -K * H;
    tmpP1 += Eigen::MatrixXd::Identity(n, n);
    tmpP0.noalias() = tmpP1 * P;

    // Known as "Joseph form", this is valid for any gain K. Additionally it may be less
    // sensitive to numerical instability, but in our algorithm it seemed to make no difference.
    P.noalias() = tmpP0 * tmpP1.transpose();
    tmpP0.noalias() = R * K.transpose();
    P.noalias() += K * tmpP0;
}

// First half of a general KF update.
// I made these updates functions rather than methods in order to reduce bugs where
// a matrix defined in a class method accidentally shadows one of
// the class member matrices (used for efficiency) and then this part uses
// the wrong, unshadowed variable.
void update(Eigen::VectorXd& m, Eigen::MatrixXd& P, const Eigen::VectorXd& y,
            const Eigen::MatrixXd& H, const Eigen::MatrixXd& R, Eigen::MatrixXd& K,
            Eigen::MatrixXd& HP, Eigen::MatrixXd& tmpP0,
            Eigen::LDLT<Eigen::MatrixXd> &invS)
{
    // Support a truncated representation where H has less columns than
    // it "should" and the rest of the columns are assumed to be zeros,
    // which is more efficient to compute with
    const int l = H.cols();
    assert(l <= P.rows());

    HP.noalias() = H * P.topRows(l);

    Eigen::MatrixXd &S = tmpP0;
    S = R;
    S.noalias() += HP.leftCols(l) * H.transpose();
    invS.compute(S);
    tmpP0 = invS.solve(HP);
    K = tmpP0.transpose();

    Eigen::MatrixXd &v = tmpP0;
    v.noalias() = -H * m.topRows(l);
    v += y;
    m.noalias() += K * v; // m += K * (y - H * m)
    updateCommon(m, P, HP, K);
}

inline double pow2(double x) {
    return x*x;
}

struct EKFImplementation : public EKF {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Scaling noise values is a standard technique for making KFs more numerically robust,
    // but the implementation may still have issues. Use with care (set to 1 by default)
    const double noiseScale;

    Eigen::Vector3d gravity;

    // KF state (stateDim).
    Eigen::VectorXd m;
    // KF state covariance (stateDim * stateDim).
    Eigen::MatrixXd P;
    // KF process noise covariance.
    Eigen::Matrix<double, Q_DIM, Q_DIM> Q;

    // Reuse large matrices.
    // Temporary matrix in KF predict
    Eigen::Matrix<double, INER_DIM, INER_DIM> dydx;
    // Temporary matrix in KF predict (stateDim * 6).
    Eigen::Matrix<double, INER_DIM, Q_DIM> dydq;
    // Key matrix in KF update (n * stateDim, n = 1,2,3,4). Do not use this as argument name to methods to avoid shadowing.
    Eigen::MatrixXd H;
    // Temporary matrix in KF update (n * stateDim).
    Eigen::MatrixXd HP;
    // Temporary matrix in KF update (stateDim * n).
    Eigen::MatrixXd K;
    // Temporary matrix in KF update (stateDim * stateDim).
    Eigen::MatrixXd tmpP1;
    // Temporary matrix in KF update (stateDim * stateDim).
    Eigen::MatrixXd tmpP0;
    // Temporary matrix in KF update.
    Eigen::MatrixXd S;
    // Temporary matrix decomposition in the KF update. Represents the inverse of S
    Eigen::LDLT<Eigen::MatrixXd> invS;
    // Temporary matrix in KF update (n * n). Do not use this as argument name to methods to avoid shadowing
    Eigen::MatrixXd R;

    // Visual update matrices.
    // Using sparse matrices is an option, but prefer using dense matrices
    // while developing to avoid issues.
    Eigen::SparseMatrix<double> visAugH;
    // Matrix at list element i is an augmentation operation that drops
    // camera pose i from the trail (where 0 is the first historical pose)
    std::vector<Eigen::SparseMatrix<double>> visAugA;
    Eigen::SparseMatrix<double> visAugQ;
    Eigen::SparseMatrix<double> visUnaugmentA;

    // Temporary matrix for pose trail rotations (from SLAM)
    Eigen::MatrixXd trailRotationA;

    const Parameters &parameters;

    const int camPoseCount;
    const int hybridMapDim;
    const int stateDim;

    int augmentCount;
    std::vector<double> augmentTimes;
    double time, ZUPTtime, ZRUPTtime, initZUPTtime;
    bool wasStationary;
    double prevSampleT;
    double firstSampleT;
    bool firstSample;

    EKFImplementation(const Parameters &parameters) :
        noiseScale(parameters.odometry.noiseScale * parameters.odometry.noiseScale),
        parameters(parameters),
        camPoseCount(parameters.odometry.cameraTrailLength),
        hybridMapDim(parameters.odometry.hybridMapSize * MAP_POINT_DIM),
        stateDim(INER_DIM + camPoseCount * POSE_DIM + hybridMapDim),
        augmentCount(0),
        augmentTimes(),
        time(0.0),
        ZUPTtime(-1.0),
        ZRUPTtime(-1.0),
        initZUPTtime(-1.0),
        wasStationary(false),
        prevSampleT(-1.0),
        firstSampleT(-1.0),
        firstSample(true)
    {
        const ParametersOdometry& po = parameters.odometry;

        gravity << 0, 0, -po.gravity;

        m = Eigen::MatrixXd::Zero(stateDim, 1);
        P = Eigen::MatrixXd::Zero(stateDim, stateDim);

        // while it's allowed in theory to have more rows than stateDim,
        // such updates are probably best avoided in practice. They could occur
        // with batch visual updates, depending on the batching parameters
        const int maxHRows = stateDim;

        // preallocate all temporary matrices
        H = Eigen::MatrixXd::Zero(maxHRows, stateDim);
        HP = H;
        K = Eigen::MatrixXd::Zero(maxHRows, maxHRows);
        tmpP0 = P;
        tmpP1 = P;
        S = K;
        R = K;
        invS = Eigen::LDLT<Eigen::MatrixXd>(maxHRows);

        dydx.setZero();
        dydq.setZero();

        // Set placeholder value for the orientation. The orientation
        // should be initialized separately.
        m.segment(ORI, 4) << 1, 0, 0, 0;

        m.segment(BAT, 3) << 1.0, 1.0, 1.0;

        P.block(POS, POS, 3, 3).setIdentity(3, 3) *= pow2(po.noiseInitialPos);
        P.block(VEL, VEL, 3, 3).setIdentity(3, 3) *= pow2(po.noiseInitialVel);
        P.block(ORI, ORI, 4, 4).setIdentity(4, 4); // Placeholder.

        P.block(BGA, BGA, 3, 3).setIdentity(3, 3) *= pow2(po.noiseInitialBGA);
        P.block(BAA, BAA, 3, 3).setIdentity(3, 3) *= pow2(po.noiseInitialBAA);
        P.block(BAT, BAT, 3, 3).setIdentity(3, 3) *= pow2(po.noiseInitialBAT);

        P(SFT, SFT) = pow2(po.noiseInitialSFT);

        const double noisePos = pow2(po.noiseInitialPosTrail);
        const double noiseOrientation = pow2(po.noiseInitialOriTrail);
        for (int i = 0; i < camPoseCount; i++) {
            int m = CAM + i * POSE_DIM;
            P.block(m, m, 3, 3).setIdentity(3, 3) *= noisePos;
            P.block(m + 3, m + 3, 4, 4).setIdentity(4, 4) *= noiseOrientation;
        }

        // Process noise covariance matrix Q.
        Q = Eigen::MatrixXd::Zero(Q_DIM, Q_DIM);
        Q.block(Q_ACC, Q_ACC, 3, 3).setIdentity(3, 3) *= pow2(po.noiseProcessAcc);
        Q.block(Q_GYRO, Q_GYRO, 3, 3).setIdentity(3, 3) *= pow2(po.noiseProcessGyro);

        P *= noiseScale;
        Q *= noiseScale;

        const int poseTrailDim = stateDim - hybridMapDim;

        // Prepare constant matrices used in visual update and augmentation.
        for (int droppedPoseIndex = 0; droppedPoseIndex < camPoseCount; ++droppedPoseIndex) {
            visAugA.push_back(Eigen::SparseMatrix<double>(stateDim, stateDim));
            Eigen::SparseMatrix<double> &A = visAugA.back();
            for (int i = 0; i < CAM; i++) {
                // Don't change main state.
                A.insert(i, i) = 1;
            }
            for (int i = CAM; i < CAM + droppedPoseIndex * POSE_DIM; ++i) {
                // Shift poses by one, until "droppedPoseIndex", which is dropped...
                assert(i + POSE_DIM < stateDim);
                A.insert(i + POSE_DIM, i) = 1;
            }

            // ... and don't change the rest of the state
            for (int i = CAM + (droppedPoseIndex + 1) * POSE_DIM; i < stateDim; i++) {
                A.insert(i, i) = 1;
            }
            A.makeCompressed();
        }

        // "Undo augmentation" matrix
        visUnaugmentA = Eigen::SparseMatrix<double>(stateDim, stateDim);

        for (int i = 0; i < CAM; i++) {
            // Don't change main state.
            visUnaugmentA.insert(i, i) = 1;
        }
        for (int i = CAM; i + POSE_DIM < poseTrailDim; i++) {
            // Shift poses by one, dropping the first one and replacing
            // the last one (if present) with zeros
            visUnaugmentA.insert(i, i + POSE_DIM) = 1;
        }
        for (int i = poseTrailDim; i < stateDim; ++i) {
            // Don't change hybrid map
            visUnaugmentA.insert(i, i) = 1;
        }

        visAugH = Eigen::SparseMatrix<double>(POSE_DIM, stateDim);
        for (int i = 0; i < 3; i++) {
            // Match new pose position to main state position.
            visAugH.insert(i, POS + i) = 1;
            visAugH.insert(i, CAM + i) = -1;
        }
        for (int i = 0; i < 4; i++) {
            // Match new pose orientation to main state orientation.
            visAugH.insert(3 + i, ORI + i) = 1;
            visAugH.insert(3 + i, CAM + 3 + i) = -1;
        }

        // Noise corresponding to the first augmented pose.
        visAugQ = Eigen::SparseMatrix<double>(stateDim, stateDim);
        for (int i = CAM; i < CAM + 3; i++) {
            visAugQ.insert(i, i) = noisePos;
        }
        for (int i = CAM + 3; i < CAM + POSE_DIM; i++) {
            visAugQ.insert(i, i) = noiseOrientation;
        }
        visAugQ *= noiseScale;

        visAugH.makeCompressed();
        visAugQ.makeCompressed();
        visUnaugmentA.makeCompressed();

        #ifdef EIGEN_RUNTIME_NO_MALLOC
        Eigen::internal::set_is_malloc_allowed(false);
        #endif
    }

    // Initialize odometry orientation from an accelerometer sample.
    void initializeOrientation(const Eigen::Vector3d &xa) {
        const ParametersOdometry& po = parameters.odometry;
        Eigen::Quaterniond qq = Eigen::Quaterniond::FromTwoVectors(-gravity, xa);
        Eigen::Vector4d q;
        q << qq.w(), qq.x(), qq.y(), qq.z();

        m.segment(ORI, 4) = q;

        // Last component of the orientation describes the arbitrary starting direction
        // in the xy plane. The function `FromTwoVectors()` always returns 0 for that
        // component, so when we set its variance to 0 the "forward" direction becomes fixed.
        assert(q[3] == 0);
        P.block(ORI, ORI, 4, 4) <<
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 0;
        P.block(ORI, ORI, 4, 4) *= pow2(po.noiseInitialOri) * noiseScale;
    }

    // EKF prediction step.
    void predict(double t, const Eigen::Vector3d &xg, const Eigen::Vector3d &xa) {
        // Variable notation:
        // xg: gyro measurement (in device coordinates)
        // xa: acc measurement (in device coordinates)
        // bg: gyro additive bias = m.segment(BGA, 3)
        // ba: acc additive bias = m.segment(BAA, 3)
        // T: acc multiplicative bias = m.segment(BAT, 3)
        // eg: gyro additive noise (does not appear in code)
        // ea: acc additive noise (does not appear in code)
        // dt: time delta
        // x: the previous state
        // y: the new state
        // p: previous position = m.segment(POS, 3) (in global coordinates)
        // v: previous velocity = m.segment(VEL, 3) (in global coordinates)
        // q: previous quaternion= m.segment(ORI, 4)
        // dydx: Jacobian, ie the collection of y's derivatives wrt
        //       p, v, q, bg, ba, T
        // dydq: Jacobian, ie the collection of y's derivatives wrt
        //       ea, eg
        //
        // The formulas, using the variable names:
        //
        // p_new = v * dt + p
        // q_new = A * q
        // v_new = (R' * (T * xa - ba + ea) + gravity) * dt + v
        // where
        // A = exp(-0.5 * S)
        // S = S(w) 4x4 matrix, where w = xg - bg + eg
        // R = quat2rmat(q_new)
        //
        // p_new depends on p, v
        // q_new depends on q, bg, eg
        // v_new depends on v, q, bg, ba, T, eg, ea
        // bg and ba can optionally have random walk dynamics.

        timer(odometry::TIME_STATS, "KF predict");

        double dt = 0.0;
        if (!firstSample) {
            dt = t - prevSampleT;
            time = t - firstSampleT;
        }
        else {
            firstSampleT = t;
            firstSample = false;
        }
        prevSampleT = t;
        if (dt <= 0.0) {
            if (time > 0) log_debug("Skipping KF predict, dt %.2f <= 0.0.", dt);
            return;
        }

        // For clarity, zero the reused matrices even though the block assignment
        // statements below should overwrite all previously assigned values.

        // the INER_DIM*INER_DIM dydx represents just the top-left block of the
        // full matrix. Rest is identity
        dydx.setIdentity();
        dydq.setZero();

        // Mean reverting (bounded) random walk for accelerometer bias.
        //
        // Quoting Arno on why random walk model may work better than increasing initial noise covariance
        // even when the true bias is constant:
        // “
        // Because we are interested in estimating a static parameter, just increasing the initial noise
        // covariance would seem to be the right thing to do. However, the non-linearities and the
        // linearizations (EKF) we do in the model tends not to like this (probably related to the fact
        // that we try to optimize so many parameters in the beginning of a session) and somehow increasing
        // the noise scale too much seems to make things unstable. The random-walk model kind of allows the
        // parameters to more slowly seek themselves into the right regime.
        // This is kind of how I see why it can work…
        //
        // It might work (and is often used in these kind of models) to also have the walk for the gyro bias.
        // However, the nature of the gyro bias behavior (it appears at a higher level in the dynamical
        // model) is quite different from the acc bias, so there is no guarantee that it works in a same way.
        // ”
        if (parameters.odometry.noiseProcessBAA > 0.0) {
            const double qc = pow2(parameters.odometry.noiseProcessBAA);
            const double theta = parameters.odometry.noiseProcessBAARev;
            Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3) *= noiseScale * qc;
            if (theta > 0.0) {
                Q.block(Q_BAA_DRIFT, Q_BAA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
            }
        }
        if (parameters.odometry.noiseProcessBGA > 0.0) {
            const double qc = pow2(parameters.odometry.noiseProcessBGA);
            const double theta = parameters.odometry.noiseProcessBGARev;
            Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3) *= noiseScale * qc;
            if (theta > 0.0) {
                Q.block(Q_BGA_DRIFT, Q_BGA_DRIFT, 3, 3) *= (1 - exp(-2 * dt * theta)) / (2 * theta);
            }
        }

        // Gyro rotation
        const Eigen::Vector3d w = xg - m.segment(BGA, 3);
        Eigen::Matrix4d S;
        S <<
                0, -w[0], -w[1], -w[2],
                w[0], 0, -w[2], w[1],
                w[1], w[2], 0, -w[0],
                w[2], -w[1], w[0], 0;

        // Quaternion rotation from gyroscope
        S *= -dt / 2;
        Eigen::Matrix4d A = S.exp();

        // Rotation from quaternion and the derivative
        Eigen::Matrix3d dR[4];
        Eigen::Matrix3d R = odometry::util::quat2rmat_d(A * m.segment(ORI, 4), dR);

        // Position
        m.segment(POS, 3) += m.segment(VEL, 3) * dt;

        // Velocity
        Eigen::Vector3d Txab = m.segment(BAT, 3).asDiagonal() * xa - m.segment(BAA, 3);
        m.segment(VEL, 3) += (R.transpose() * Txab + gravity) * dt;

        // Orientation
        Eigen::Vector4d prevQuat = m.segment(ORI, 4);
        m.segment(ORI, 4) = A * m.segment(ORI, 4);

        // BGA and BAA mean reversion
        if (parameters.odometry.noiseProcessBAA > 0.0) {
            m.segment(BAA, 3) *= exp(-dt * parameters.odometry.noiseProcessBAARev);
        }
        if (parameters.odometry.noiseProcessBGA > 0.0) {
            m.segment(BGA, 3) *= exp(-dt * parameters.odometry.noiseProcessBGARev);
        }

        dydx.block(POS, POS, 3, 3).setIdentity(3, 3);
        dydx.block(VEL, VEL, 3, 3).setIdentity(3, 3);
        dydx.block(POS, VEL, 3, 3).setIdentity(3, 3) *= dt;
        dydx.block(BGA, BGA, 3, 3).setIdentity(3, 3);
        dydx.block(BAA, BAA, 3, 3).setIdentity(3, 3);
        dydx.block(BAT, BAT, 3, 3).setIdentity(3, 3);

        // Derivatives of the velocity w.r.t. to the quaternion
        for (int i = 0; i < 4; i++) {
            dydx.block(VEL, ORI + i, 3, 1) = dR[i].transpose() * Txab * dt;
        }
        dydx.block(VEL, ORI, 3, 4) = dydx.block(VEL, ORI, 3, 4) * A;

        // Derivatives of the quaternion w.r.t. itself
        dydx.block(ORI, ORI, 4, 4) = A;

        // Derivatives of the velocity w.r.t. acceleration noise
        dydq.block(VEL, Q_ACC, 3, 3) = R.transpose() * dt;

        // Derivatives of the quaternion w.r.t. gyroscope noise
        Eigen::Matrix4d dS0, dS1, dS2;
        dS0 << 0, dt / 2, 0, 0, -dt / 2, 0, 0, 0, 0, 0, 0, dt / 2, 0, 0, -dt / 2, 0;
        dS1 << 0, 0, dt / 2, 0, 0, 0, 0, -dt / 2, -dt / 2, 0, 0, 0, 0, dt / 2, 0, 0;
        dS2 << 0, 0, 0, dt / 2, 0, 0, dt / 2, 0, 0, -dt / 2, 0, 0, -dt / 2, 0, 0, 0;
        dydq.block(ORI, Q_GYRO, 4, 1) = A * dS0 * prevQuat;
        dydq.block(ORI, Q_GYRO + 1, 4, 1) = A * dS1 * prevQuat;
        dydq.block(ORI, Q_GYRO + 2, 4, 1) = A * dS2 * prevQuat;
        dydq.block(BGA, Q_BGA_DRIFT, 3, 3).setIdentity(3, 3);
        dydq.block(BAA, Q_BAA_DRIFT, 3, 3).setIdentity(3, 3);

        // Derivatives of the velocity w.r.t. gyroscope noise
        // It seems suspicious to me how `A` is multiplied twice in these statements.
        // Apparently the matrix exponential with inner function is difficult to
        // differentiate analytically, giving such a complex result that numerical
        // errors render it equally inaccurate. See the visualization in the `der_predict`
        // unit test.
        dydq.block(VEL, Q_GYRO, 3, 3) = dydx.block(VEL, ORI, 3, 4) * dydq.block(ORI, Q_GYRO, 4, 3);

        // Derivatives of the velocity w.r.t. to the gyro bias
        dydx.block(VEL, BGA, 3, 3) = -dydq.block(VEL, Q_GYRO, 3, 3);

        // Derivatives of the quaternion w.r.t. to the gyro bias
        dydx.block(ORI, BGA, 4, 3) = -dydq.block(ORI, Q_GYRO, 4, 3);

        // Derivatives of the velocity w.r.t the acc. bias
        dydx.block(VEL, BAA, 3, 3) = -R.transpose() * dt;

        // Derivatives of the velocity w.r.t the acc. transformation
        dydx.block(VEL, BAT, 3, 3) = R.transpose() * xa.asDiagonal() * dt;

        // The following code performs more efficiently the state covariance update
        //     P = dydx * P * dydx' + dydq * Q * dydq'
        // by exploiting the structure of `dydx` which is a block matrix with a large identity
        // block sided by two zero blocks.
        P.topLeftCorner<INER_DIM, INER_DIM>() = dydx * P.topLeftCorner<INER_DIM, INER_DIM>() * dydx.transpose() + dydq * Q * dydq.transpose();
        tmpP0.noalias() = P.leftCols<INER_DIM>().bottomRows(stateDim - INER_DIM) * dydx.transpose();
        P.block(INER_DIM, 0, stateDim - INER_DIM, INER_DIM) = tmpP0;
        tmpP0.noalias() = dydx * P.topRows<INER_DIM>().rightCols(stateDim - INER_DIM);
        P.block(0, INER_DIM, INER_DIM, stateDim - INER_DIM) = tmpP0;

        // Do not call maintainPositiveSemiDefinite here. Modifying the entire matrix P
        // could easily be the heaviest operation in this method and
        // non-symmetry should not appear so fast that correction would be
        // needed on each sample
    }

    Eigen::Vector3d position() const {
        return m.segment(POS, 3);
    }

    Eigen::Vector3d velocity() const {
        return m.segment(VEL, 3);
    }

    Eigen::Vector4d orientation() const {
        return m.segment(ORI, 4);
    }

    Eigen::Vector3d biasGyroscopeAdditive() const {
        return m.segment(BGA, 3);
    }

    Eigen::Vector3d biasAccelerometerAdditive() const {
        return m.segment(BAA, 3);
    }

    Eigen::Vector3d biasAccelerometerTransform() const {
        return m.segment(BAT, 3);
    }

    int camTrailSize() const {
        return camPoseCount;
    }

    Eigen::Vector3d historyPosition(int i) const {
        if (i == -1) return position();
        assert(i >= 0 && i < camPoseCount);
        return m.segment(CAM + POSE_DIM * i, 3);
    }

    Eigen::Vector4d historyOrientation(int i) const {
        if (i == -1) return orientation();
        assert(i >= 0 && i < camPoseCount);
        return m.segment(CAM + POSE_DIM * i + 3, 4);
    }

    double historyTime(int i) const {
        if (i == -1) return getPlatformTime();
        assert(i >= 0 && i < camPoseCount);
        size_t n = augmentTimes.size();
        assert(i < static_cast<int>(n));
        return augmentTimes[n - i - 1];
    }

    double speed() const {
        return m.segment(VEL, 3).norm();
    }

    double horizontalSpeed() const {
        return m.segment(VEL, 2).norm();
    }

    // Zero velocity update aka ZUPT. Sets velocity to zero.
    void updateZupt(double r) {
        if (time - ZUPTtime < 0.25) {
            return;
        }
        ZUPTtime = time;
        wasStationary = true;

        H = Eigen::MatrixXd::Zero(3, VEL + 3); // truncated representation
        H.block(0, VEL, 3, 3).setIdentity();
        R = Eigen::MatrixXd::Identity(3, 3) * r * noiseScale;

        // Note: for very non-trivial reasons, this is better here than
        // Eigen::Vector3d, which would cause a malloc in update
        Eigen::MatrixXd &y = tmpP1;
        y = Eigen::VectorXd::Zero(3);

        update(m, P, y, H, R, K, HP, tmpP0, invS);
    }

    // Do ZUPT where the variance is scaled by expired time since start of session.
    // Helpful for initializing tracking for datasets which lack stationarity updates.
    void updateZuptInitialization() {
        // The init ZUPT has a bad tendency to contract the positioning track, so
        // turn it off after first (visual) stationarity update because at that
        // point the odometry should be quite stable.
        if (wasStationary || time > 60 || time - initZUPTtime < 0.1) {
            return;
        }
        initZUPTtime = time;

        H = Eigen::MatrixXd::Zero(3, VEL + 3); // truncated representation
        H.block(0, VEL, 3, 3).setIdentity();
        R = Eigen::MatrixXd::Identity(3, 3) * parameters.odometry.initZuptR * noiseScale * exp(0.5 * time);

        Eigen::MatrixXd &y = tmpP1;
        y = Eigen::VectorXd::Zero(3);

        update(m, P, y, H, R, K, HP, tmpP0, invS);
    }

    // Zero rotation update aka ZRUPT. Sets gyroscope bias to given sample.
    void updateZrupt(const Eigen::Vector3d &xg) {
        if (time - ZRUPTtime < 0.25) {
            return;
        }
        ZRUPTtime = time;

        H = Eigen::MatrixXd::Zero(3, BGA + 3); // truncated representation
        H.block(0, BGA, 3, 3).setIdentity();
        R = Eigen::MatrixXd::Identity(3, 3) * parameters.odometry.rotationZuptR * noiseScale;

        update(m, P, xg, H, R, K, HP, tmpP0, invS);
    }

    // Velocity pseudo update. Brings velocity down to target speed without changing direction.
    void updatePseudoVelocity(double defaultSpeed, double r) {
        // Only update the x and y coordinates of speed so that the corrections don't
        // bleed into the vertical component and so wreck the vertical position estimate.
        double h = m.segment(VEL, 2).norm();
        R = Eigen::MatrixXd::Identity(1, 1) * r * noiseScale;

        H = Eigen::RowVectorXd::Zero(VEL + 2); // truncated representation
        if (h <= 1e-7) {
            return;
        }
        for (int i = 0; i < 2; i++) {
            H(0, VEL + i) = m[VEL + i] / h;
        }

        const int l = H.cols();
        HP.noalias() = H * P.topRows(l);
        S.noalias() = HP.leftCols(l) * H.transpose();
        double s = S(0, 0) + R(0, 0);
        K = HP.transpose() / s;
        m.noalias() += K * (defaultSpeed - h);
        updateCommon(m, P, HP, K);
    }

    void updatePosition(const Eigen::Vector3d &y, double r) {
        R = Eigen::MatrixXd::Identity(3, 3) * r * noiseScale;
        H = Eigen::MatrixXd::Zero(3, POS + 3); // truncated representation
        H.block(0, POS, 3, 3).setIdentity();
        update(m, P, y, H, R, K, HP, tmpP0, invS);
        maintainPositiveSemiDefinite();
    }

    // Update the position z component (height) to zero.
    void updateZeroHeight(double r) {
        H = Eigen::MatrixXd::Zero(1, POS + 3); // truncated representation
        H(POS + 2) = 1;
        R = Eigen::MatrixXd::Identity(1, 1) * r * noiseScale;
        Eigen::Matrix<double, 1, 1> y;
        y << 0;
        update(m, P, y, H, R, K, HP, tmpP0, invS);
        maintainPositiveSemiDefinite();
    }

    void updateOrientation(const Eigen::Vector4d &q, double r) {
        H = Eigen::MatrixXd::Zero(4, ORI + 4); // truncated representation
        H.block(0, ORI, 4, 4).setIdentity();
        R = Eigen::MatrixXd::Identity(4, 4) * r * noiseScale;
        update(m, P, q, H, R, K, HP, tmpP0, invS);
        normalizeQuaternions();
        maintainPositiveSemiDefinite();
    }

    void getInertialState(VectorInertialMean &mean, MatrixInertialCov &cov) const final {
        mean = m.segment<INER_DIM>(0);
        cov = P.topLeftCorner<INER_DIM, INER_DIM>();
    }

    void setInertialState(const VectorInertialMean &mean, const MatrixInertialCov &cov) final {
        m.segment<INER_DIM>(0) = mean;
        P.topLeftCorner<INER_DIM, INER_DIM>() = cov;
        // mark pose trail as invalid
        augmentCount = 0;
        augmentTimes.clear();
    }

    double getImuToCameraTimeShift() const {
        return m(SFT);
    }

    void translateTo(const Eigen::Vector3d &pos) final {
        const Eigen::Vector3d deltaX = pos - position();
        m.segment(POS, 3) += deltaX;
        for (int i = 0; i < camPoseCount; ++i) {
            m.segment(CAM + POSE_DIM * i, 3) += deltaX;
        }
    }

    void transformTo(const Eigen::Vector3d &pos, const Eigen::Vector4d &q, int i) final {
        // Coordinate transform for quaternions
        Eigen::Matrix4d qChangeMat;
        // Coordinate rotation for position & velocity vectors
        Eigen::Matrix3d pChangeMat;
        // Translation for position vectors
        Eigen::Vector3d translation;

        // compute the above three in this block
        {
            // wrangle between different representations
            const Eigen::Vector4d q0 = i < 0 ? orientation() : historyOrientation(i);
            const Eigen::Quaterniond quat0(q0(0), q0(1), q0(2), q0(3));
            const Eigen::Quaterniond quat1(q(0), q(1), q(2), q(3));

            // Assuming normalized quaternions.
            // Note that qChange right-multiples the quaternion
            const Eigen::Quaterniond qChange = quat0.conjugate() * quat1;

            // Quaternion right-multiplication as a rotation matrix. See, e.g.,
            // https://users.aalto.fi/~ssarkka/pub/quat.pdf
            const double p1 = qChange.w(), p2 = qChange.x(), p3 = qChange.y(), p4 = qChange.z();
            qChangeMat <<
                p1, -p2, -p3, -p4,
                p2, p1, p4, -p3,
                p3, -p4, p1, p2,
                p4, p3, -p2, p1;

            // note: invert/transpose for pos & velocity transforms
            pChangeMat = qChange.toRotationMatrix().transpose();

            const Eigen::Vector3d refPos = (i < 0 ? position() : historyPosition(i));
            translation = pos - pChangeMat * refPos;
        }

        // And apply them in a Linear Kalman-prediction-like update
        trailRotationA = Eigen::MatrixXd::Identity(stateDim, stateDim);
        trailRotationA.block<3, 3>(POS, POS) = pChangeMat;
        trailRotationA.block<3, 3>(VEL, VEL) = pChangeMat;
        trailRotationA.block<4, 4>(ORI, ORI) = qChangeMat;
        for (int i = 0; i < camPoseCount; ++i) {
            const int poseOffs = CAM + i * POSE_DIM;
            trailRotationA.block<3, 3>(poseOffs, poseOffs) = pChangeMat;
            trailRotationA.block<4, 4>(poseOffs + 3, poseOffs + 3) = qChangeMat;
        }

        // no noise or quaternion normalization required. Involves only exact
        // rotations that preserve the quaternion norms
        tmpP0.noalias() = trailRotationA * m;
        m = tmpP0;
        tmpP0.noalias() = P * trailRotationA.transpose();
        P.noalias() = trailRotationA * tmpP0;

        translateTo(position() + translation);
    }

    void visualTrackUpdateCommon(
        const Eigen::MatrixXd &visH,
        const Eigen::VectorXd &f,
        const Eigen::VectorXd &y,
        double r)
    {
        assert(!f.hasNaN());
        assert(!y.hasNaN());
        assert(!visH.hasNaN());
        const int l = static_cast<int>(visH.cols());
        const int n = static_cast<int>(visH.rows());
        assert(l <= P.rows());
        assert(l > 0);
        assert(n > 0);
        assert(y.size() == n);
        assert(f.size() == n);

        R.setIdentity(n, n) *= (r * r) * noiseScale;

        HP.noalias() = visH * P.topRows(l);
        S.noalias() = HP.leftCols(l) * visH.transpose();
        S += R;
        invS.compute(S);
    }

    // Check if a proposed visual track update passes the chi2 outlier test
    // and an RMSE threshold check.
    VuOutlierStatus visualTrackOutlierCheck(
            const Eigen::MatrixXd &visH,
            const Eigen::VectorXd &f,
            const Eigen::VectorXd &y,
            double r,
            double trackRmseThreshold) {

        int n = static_cast<int>(visH.rows());
        const Eigen::VectorXd v = y - f;

        // fail the RMSE check early (no need to compute the heavier check)
        if (trackRmseThreshold >= 0.0) {
            const double rmse = sqrt((v.dot(v)) / n);
            if (rmse > trackRmseThreshold) return VuOutlierStatus::RMSE;
        }

        if (r < 0.0) return VuOutlierStatus::INLIER;
        visualTrackUpdateCommon(visH, f, y, r); // Compute `invS`.

        assert((camPoseCount + 1) * 2 < static_cast<int>(chi2inv95.size())); // Max value of n.
        assert(n < static_cast<int>(chi2inv95.size()));

        // Otto: I added noiseScale here based on this hand-wavy analysis:
        // in updateVisualTrack, noiseScale does not affect the change in m,
        // since the Kalman gain is computed as (S^-1 * H * P)^T, where the
        // noiseScale parmeter effectively multiplies both S and P so it
        // cancels out in the multiplication. The multiplication of P is by
        // noiseScale is missing in v so it needs to be added separately.
        const double t = noiseScale * invS.solve(v).transpose() * v;
        if (t > chi2inv95[n]) return VuOutlierStatus::CHI2;

        return VuOutlierStatus::INLIER;
    }

    // Visual update.
    // The first part of the visual update procedure is in prepareVisualUpdate(),
    // here is mostly just the final KF update matrix manipulation.
    //
    // Given predicted (triangulated & projected) visual track `f`,
    // measured (tracker output) track `y`, and Jacobian `visH` of the measurement function
    // y = h(), do KF update. If the track `f` is recomputed using the updated state, then
    // it should be closer to the track `y` than before (the RMSE value goes down).
    void updateVisualTrack(
            const Eigen::MatrixXd &visH,
            const Eigen::VectorXd &f,
            const Eigen::VectorXd &y,
            double r)
    {
        visualTrackUpdateCommon(visH, f, y, r);
        K = invS.solve(HP).transpose();
        Eigen::MatrixXd &v = tmpP0;
        v = y - f;
        m.noalias() += K * v;
        P.noalias() -= K * HP;

        // maintainPositiveSemiDefinite();
        normalizeQuaternions();
    }

    // KF "update trick" which clones the current pose, rearranges the existing
    // pose trail (mostly moves it one slot forward) and drops one pose
    void updateVisualPoseAugmentation(int discardedPoseIndex) {
        if (discardedPoseIndex == -1) discardedPoseIndex = camPoseCount - 1;
        Eigen::SparseMatrix<double> &A = visAugA.at(discardedPoseIndex);

        // Extra prediction step.
        tmpP0.noalias() = A * m;
        m = tmpP0;
        tmpP1.noalias() = P * A.transpose();
        tmpP0.noalias() = A * tmpP1;
        P = tmpP0 + visAugQ;

        // do not use setIdentity here (causes malloc for magical reasons)
        R = Eigen::MatrixXd::Identity(POSE_DIM, POSE_DIM) * parameters.odometry.augmentR * noiseScale;

        HP.noalias() = visAugH * P;
        S = R;
        S.noalias() += HP * visAugH.transpose();
        invS.compute(S);
        K = invS.solve(HP).transpose();
        Eigen::MatrixXd &v = tmpP0;
        v.noalias() = -visAugH * m;
        m.noalias() += K * v; // m += K * (-visAugH * m)

        // Use Joseph form, seems to affect results.
        updateCommonJosephForm(P, visAugH, R, K, tmpP0, tmpP1);
        maintainPositiveSemiDefinite();
        normalizeQuaternions();

        augmentTimes.push_back(getPlatformTime());
        if (augmentCount < camPoseCount) {
            // Track how many trailing poses are available.
            augmentCount++;
        }
        else {
            augmentTimes.erase(augmentTimes.begin());
        }
        assert(static_cast<int>(augmentTimes.size()) == augmentCount);
    }

    // Drops the first augmented pose. Required to handle non-keyframes correctly
    void updateUndoAugmentation() {
        tmpP0.noalias() = visUnaugmentA * m;
        m = tmpP0;

        // no need to add any noise, except for the last element of the pose
        // trail, which is ignored by the visual update anyway, therefore Q = 0
        tmpP0.noalias() = P * visUnaugmentA.transpose();
        P.noalias() = visUnaugmentA * tmpP0; // P = A P A^T

        // skipping maintainPositiveSemiDefinite and normalization

        assert(augmentCount > 0);
        augmentTimes.pop_back();
        augmentCount--;
        assert(static_cast<int>(augmentTimes.size()) == augmentCount);
    }

    Eigen::Vector3d getMapPoint(int idx) const final {
        const int offset = getMapPointStateIndex(idx);
        assert(idx >= 0 && offset + MAP_POINT_DIM <= stateDim);
        return m.segment<3>(offset);
    }

    void insertMapPoint(int idx, const Eigen::Vector3d &pf) final {
        const int offset = getMapPointStateIndex(idx);
        assert(idx >= 0 && offset + MAP_POINT_DIM <= stateDim);
        P.block(offset, 0, MAP_POINT_DIM, stateDim).setZero();
        P.block(0, offset, stateDim, MAP_POINT_DIM).setZero();
        m.segment<MAP_POINT_DIM>(offset).setZero();

        constexpr double NOISE = 1e3;
        P.block<MAP_POINT_DIM, MAP_POINT_DIM>(offset, offset).setIdentity() *= (NOISE*NOISE);
        m.segment<MAP_POINT_DIM>(offset) = pf;
    }

    int getMapPointStateIndex(int idx) const final {
        if (idx == -1) return -1;
        return stateDim - hybridMapDim + idx * MAP_POINT_DIM;
    }

    void conditionOnLastPose() {
        assert(hybridMapDim == 0);
        assert(augmentCount > 0);

        const int m = stateDim - POSE_DIM;
        P.block(0, 0, m, m) -= P.block(0, m, m, POSE_DIM) *
            P.block<POSE_DIM, POSE_DIM>(m, m).inverse() *
            P.block(m, 0, POSE_DIM, m);

        P.block(0, m, m, POSE_DIM).setZero();
        P.block(m, 0, POSE_DIM, m).setZero();

        constexpr double NOISE = 1e3;
        P.block<POSE_DIM, POSE_DIM>(m, m).setIdentity() *= (NOISE*NOISE);
    }

    void lockBiases() {
        P.block(BGA, 0, 9, stateDim).setZero();
        P.block(0, BGA, stateDim, 9).setZero();
    }

    // Getters and setters for debugging. Used by test/{triangulation.cpp, odometry.cpp}.
    void setState(const Eigen::VectorXd& _m) {
        assert(_m.size() == stateDim);
        m = _m;
    }
    void setStateCovariance(const Eigen::MatrixXd& _P) {
        assert(_P.rows() == stateDim);
        assert(_P.cols() == stateDim);
        P = _P;
    }
    void setProcessNoise(const Eigen::MatrixXd& _Q) {
        assert(_Q.rows() == Q.rows());
        assert(_Q.cols() == Q.cols());
        Q = _Q;
    }
    double getPlatformTime() const {
        return firstSampleT + time;
    }
    int getPoseCount() const {
        return augmentCount + 1;
    }
    int getStateDim() const {
        return stateDim;
    }
    const Eigen::VectorXd &getState() const {
        return m;
    }
    Eigen::MatrixXd getStateCovariance() const {
        return P;
    }
    const Eigen::MatrixXd& getStateCovarianceRef() const {
        return P;
    }
    Eigen::MatrixXd getVisAugH() const {
        return visAugH;
    }
    Eigen::MatrixXd getVisAugA() const {
        return visAugA.back();
    }
    Eigen::MatrixXd getVisAugQ() const {
        return visAugQ;
    }
    Eigen::MatrixXd getDydx() const {
        Eigen::MatrixXd full = Eigen::MatrixXd::Identity(P.rows(), P.cols());
        full.topLeftCorner<INER_DIM, INER_DIM>() = dydx;
        return full;
    }

    // Print key information of the KF state and covariance inertial part.
    std::string stateAsString() const {
        std::stringstream ss;
        Eigen::Matrix<double, INER_DIM, 1> var = P.block(0, 0, INER_DIM, INER_DIM).diagonal();
        for (size_t i = 0; i < STATE_PARTS.size(); i++) {
            int part = STATE_PARTS[i];
            int part_size = STATE_PART_SIZES[i];
            ss << STATE_PART_NAMES[i] << " ";
            bool negative_variance = false;
            for (int j = 0; j < part_size; j++) {
                ss << std::setprecision(3) << m(part + j) << " ";
                if (var(part + j) < 0.0) {
                    negative_variance = true;
                }
            }
            const double v = sqrt(var.segment(part, part_size).maxCoeff());
            ss << std::setprecision(2);
            ss << " [" << v << "], ";
            if (i == 2) {
                ss << std::endl << " ";
            }
        }
        ss << std::fixed << std::setprecision(3);
        ss << "t " << time;
        return ss.str();
    }

    void normalizeQuaternions(bool onlyCurrent = false) {
        m.segment(ORI, 4).normalize();
        if (onlyCurrent) return;
        for (int i = 0; i < camPoseCount; i++) {
            // If the segment is just zeroes (before first augments), then the normalized
            // segment becomes also just zeroes.
            m.segment(CAM + POSE_DIM * i + 3, 4).normalize();
        }
    }

    // For tests doing a single predict() call.
    void setFirstSampleTime(double t) {
        assert(t > 0.0);
        firstSample = false;
        firstSampleT = t;
        prevSampleT = t;
        time = t;
    }

    // Expensive, use only for debugging.
    bool isPositiveSemiDefinite() {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(P);
        if (eigensolver.info() != Eigen::Success) {
            log_warn("isPositiveSemiDefinite(): EigenSolver failed.");
            return false;
        }
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        for (int i = 0; i < eigenvalues.size(); i++) {
            if (eigenvalues[i] < 0.0) {
                return false;
            }
        }
        return true;
    }

    void maintainPositiveSemiDefinite() {
        // In theory, P is always symmetric unless there are bugs in the
        // implementation. However, numerical errors can add non-symmetry
        // that starts to accumulate and (usually very slowly) becomes
        // significant with time. It may be good to symmetrize the matrix
        // again from time to time (not on each sample / prediction) as follows
        tmpP0.noalias() = 0.5 * (P + P.transpose());
        P.swap(tmpP0);
    }

    bool getWasStationary() const {
        return wasStationary;
    }

    // for tests
    std::unique_ptr<EKF> clone() const {
        #ifdef EIGEN_RUNTIME_NO_MALLOC
        Eigen::internal::set_is_malloc_allowed(true);
        #endif
        return std::unique_ptr<EKF>(new EKFImplementation(*this));
    }
};
} // anonymous namespace

namespace odometry {
EKF::~EKF() = default;
EKF::EKF(const EKF &other) = default;

std::unique_ptr<EKF> EKF::build(const Parameters &parameters) {
    #ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
    #endif
    return std::unique_ptr<EKF>(new EKFImplementation(parameters));
}

EKF::EKF() {}
}
