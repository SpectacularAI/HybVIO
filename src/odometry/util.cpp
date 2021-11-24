#include <cmath>
#include <Eigen/Geometry>
#include "util.hpp"
#include <iostream>

namespace odometry {
namespace util {

// This is the same function as quat2rotm() from Matlab Robotics toolbox.
Eigen::Matrix3d quat2rmat(const Eigen::Vector4d& q) {
    // Note that it is also possible to compute this with Eigen as follows
    //
    //      Eigen::Quaterniond q(qvec(0), qvec(1), qvec(2), qvec(3));
    //      return q.toRotationMatrix();
    //
    // However, having the formula expanded here is good in the sense that
    // we also compute the derivative quat2rmat_d and this ensures that the
    // same formula is used in both.

    Eigen::Matrix3d R;
    R <<
        q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2],
        2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1],
        2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
    return R;
}

// Derivatives of the rotation matrix w.r.t. the quaternion of the quat2rmat() function.
Eigen::Matrix3d quat2rmat_d(const Eigen::Vector4d& q, Eigen::Matrix3d(&dR)[4]) {
    dR[0] <<
        2*q(0), -2*q(3),  2*q(2),
        2*q(3),  2*q(0), -2*q(1),
        -2*q(2),  2*q(1),  2*q(0);
    dR[1] <<
        2*q(1),  2*q(2),  2*q(3),
        2*q(2), -2*q(1), -2*q(0),
        2*q(3),  2*q(0), -2*q(1);
    dR[2] <<
        -2*q(2),  2*q(1),  2*q(0),
        2*q(1),  2*q(2),  2*q(3),
        -2*q(0),  2*q(3), -2*q(2);
    dR[3] <<
        -2*q(3), -2*q(0),  2*q(1),
        2*q(0), -2*q(3),  2*q(2),
        2*q(1),  2*q(2),  2*q(3);
    return quat2rmat(q);
}

Eigen::Vector4d rmat2quat(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond qd(R);
    Eigen::Vector4d q; q << qd.w(), qd.x(), qd.y(), qd.z();
    return q;
}

double sgn(double val) {
    if (val < 0) return -1.0;
    return 1.0;
}

// Compute standard deviation using the unbiased estimator (normalizing by (n - 1)).
// Accepts column and row vectors, both statically and dynamically sized.
double stdev(const Eigen::Ref<const Eigen::MatrixXd>& v) {
    assert(v.rows() == 1 || v.cols() == 1);
    assert(v.rows() != 0 && v.cols() != 0);
    if (v.rows() == 1 && v.cols() == 1) {
        return 0.0;
    }
    else {
        double n = static_cast<double>(std::max(v.rows(), v.cols()) - 1.0);
        return std::sqrt((v.array() - v.mean()).square().sum() / n);
    }
}

Eigen::Matrix3d removeRotationMatrixZTilt(const Eigen::Matrix3d &origR) {
    using Eigen::Vector3d;
    using Eigen::Matrix3d;

    // NOTE, Otto: This method does remove the Z-axis tilt but I have not
    // checked if the XY-rotation part it leaves is optimal in some sense.
    // Is it true that U = argmin(|R - U(theta)| for theta in [0, 2 PI]), in
    // the sense of some matrix norm, where R is the original rotation and
    //
    //              [cos(t), -sin(t), 0]
    //       U(t) = [sin(t), cos(t),  0]
    //              [0,         0,    1]
    //
    // is an XY rotation matrix, i.e., a rotation about the Z-axis by angle t.

    // determine XY rotation...
    const Vector3d axisX(1, 0, 0);
    const Vector3d rotatedX = origR * axisX;
    const double rotationAngle = std::atan2(rotatedX.y(), rotatedX.x());

    // and only use that
    Matrix3d xyRot;
    xyRot = Eigen::AngleAxisd(rotationAngle, Vector3d(0, 0, 1));
    return xyRot;
}

Eigen::Matrix4d replacePoseOrientationKeepPosition(const Eigen::Matrix4d &poseCW, const Eigen::Matrix3d &newOrientationCW) {
    // original: poseCW = (R_CW, r_CW) = (R_WC^T, -R_WC^T r_WC)
    // replaced: poseCW' = (P_CW, p_CW) = (P_WC^T, -P_WC^T p_WC)

    // do not move camera position: r_WC = r_Camera_to_World = p_WC
    // => -R_CW^T r_CW = -P_CW^T p_CW => p_CW = P_CW R_CW^T r_CW

    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat.topLeftCorner<3, 3>() = newOrientationCW;
    mat.block<3, 1>(0, 3) = newOrientationCW
        * poseCW.topLeftCorner<3, 3>().transpose()
        * poseCW.block<3, 1>(0, 3);

    return mat;

    // altenative
    // Eigen::Matrix4d mat = poseCW.inverse();
    // mat.topLeftCorner<3, 3>() = newOrientationCW.inverse();
    // return mat.inverse();
}

void toOdometryPose(
    const Eigen::Matrix4d &worldToCamera,
    Eigen::Vector3d &position,
    Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera
) {
    const Eigen::Matrix4d imuToWorld = worldToCamera.inverse() * imuToCamera;
    position = imuToWorld.block<3, 1>(0, 3);
    orientation = odometry::util::rmat2quat(imuToWorld.topLeftCorner<3, 3>().transpose());
}

Eigen::Matrix4d toWorldToCamera(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera
) {
    const Eigen::Matrix3d rot = odometry::util::quat2rmat(orientation);
    Eigen::Matrix4d worldToImu = Eigen::Matrix4d::Identity();
    worldToImu.topLeftCorner<3, 3>() = rot;
    worldToImu.block<3, 1>(0, 3) = -rot * position;
    return imuToCamera * worldToImu;
}

Eigen::Matrix4d toCameraToWorld(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &orientation,
    const Eigen::Matrix4d &imuToCamera) {

    const Eigen::Matrix3d rot = odometry::util::quat2rmat(orientation);
    Eigen::Matrix4d imuToWorld = Eigen::Matrix4d::Identity();
    imuToWorld.topLeftCorner<3, 3>() = rot.transpose();
    imuToWorld.block<3, 1>(0, 3) = position;

    // Note: this method could be more efficient if camera-to-IMU was
    // pre-computed and used as the last parameter instead of imuToCamera
    // but this is probably not the main bottleneck
    return imuToWorld * imuToCamera.inverse();
}

// From <https://forum.kde.org/viewtopic.php?f=74&t=117430>.
double cond(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    return svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
}

double rcond(const Eigen::MatrixXd& A) {
    Eigen::FullPivLU<Eigen::MatrixXd> f(A);
    return f.rcond();
}

double rcond_ldlt(const Eigen::MatrixXd& A) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    return ldlt.rcond();
}

Eigen::MatrixXd cov2corr(const Eigen::MatrixXd& P) {
    Eigen::VectorXd d = P.diagonal();
    d = d.array().pow(-0.5).matrix();
    return d.asDiagonal() * P * d.asDiagonal();
}

void ThroughputCounter::put(double t) {
    buffer.put(t);
}

float ThroughputCounter::throughputPerSecond() {
    if (buffer.entries() < 2) {
        return 0.0;
    }
    double dur = buffer.head() - buffer.tail();
    if (dur <= 0.0) {
        return 0.0;
    }
    return ((double)buffer.entries() - 1.0) / dur;
}

} // namespace util
} // namespace odometry
