#include "triangulation.hpp"
#include "parameters.hpp"
#include "ekf.hpp"
#include "debug.hpp"

#include "../util/logging.hpp"
#include "../tracker/camera.hpp"

namespace {
using Matrix32 = odometry::Triangulator::Matrix32;
using Matrix23 = odometry::Triangulator::Matrix23;

inline double pow2(double x) {
    return x*x;
}

Eigen::Vector3f inverseToGlobal(
    const Eigen::Vector3d &pfi,
    const Eigen::Vector3d &p0,
    const Eigen::Matrix3d &R0T
) {
    const Eigen::Vector3d pfiab(pfi(0), pfi(1), 1.0);
    return ((1 / pfi(2)) * R0T * pfiab + p0).cast<float>();
}

inline Eigen::Matrix3d sumSecondDerivative(const Eigen::Matrix3d (&dd)[3], const Eigen::Vector3d &w) {
    Eigen::Matrix3d r = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) r += dd[i] * w(i);
    return r;
}

// Derivative of the pseudo-inverse
// A = matrix
// iA = pseudo-inverse of A
// dA = derivative on A w.r.t. an arbitrary variable
// The returned matrix is the derivative of the pseudoinverse w.r.t. the same
// variable as A
Matrix23 dpinv(const Matrix32 &A, const Matrix23 &iA, const Matrix32 &dA) {
    // Here is the formula for a matrix of constant rank (equation (4.12), in the Golub paper):
    //   The Differentiation of Pseudo-Inverses and Nonlinear Least Squares
    //   Problems Whose Variables Separate. Author(s): G. H. Golub and V.
    //   Pereyra. Source: SIAM Journal on Numerical Analysis, Vol. 10,
    //   No. 2 (Apr., 1973), pp. 413-432
    // See http://mathoverflow.net/questions/25778/analytical-formula-for-numerical-derivative-of-the-matrix-pseudo-inverse
    const Matrix32 iAT = iA.transpose();
    const Eigen::Matrix3d eye3 = Eigen::Matrix3d::Identity();
    const Eigen::Matrix2d eye2 = Eigen::Matrix2d::Identity();
    const Matrix23 dAT = dA.transpose();
    // MATLAB: diA = -iA*dA*iA + (iA*iA')*dA'*(eye(3)-A*iA) + (eye(2)-iA*A)*dA'*(iA'*iA);
    return -iA*dA*iA + (iA*iAT)*dAT*(eye3-A*iA) + (eye2-iA*A)*dAT*(iAT*iA);
}

bool isBehind(const Eigen::Vector3d &pf, const odometry::CameraPoseTrail &trail) {
    for (const auto &pose : trail) {
        const Eigen::Vector3d a = pose.R * (pf - pose.p);
        if (a(2) < 0) return true;
    }
    return false;
}
}

namespace odometry {

// Extract camera poses from the EKF state.
void extractCameraPoseTrail(
    const odometry::EKF& ekf,
    const std::vector<int> &poseTrailIndex,
    const Parameters &parameters,
    bool useStereo,
    CameraPoseTrail& trail)
{
    trail.clear();

    int nCameras = useStereo ? 2 : 1;
    for (int camIdx = 0; camIdx < nCameras; ++camIdx) {
        Eigen::Matrix4d imuToCam = camIdx == 0 ? parameters.imuToCamera : parameters.secondImuToCamera;
        // Check the default sentinel value has been changed.
        assert(imuToCam != Eigen::Matrix4d::Zero());
        Eigen::Matrix3d d_worldToCameraRot[4];
        const Eigen::Matrix3d imuToCameraRot = imuToCam.topLeftCorner<3, 3>();
        const Eigen::Vector3d baseline = imuToCam.block<3, 1>(0, 3);

        for (int i : poseTrailIndex) {
            const Eigen::Vector3d p = ekf.historyPosition(i - 1);
            const Eigen::Vector4d q = ekf.historyOrientation(i - 1);
            assert(!p.hasNaN() && !q.hasNaN());

            Eigen::Matrix3d worldToCameraRot = imuToCameraRot * util::quat2rmat_d(q, d_worldToCameraRot);
            CameraPose pose = {
                .p = p - worldToCameraRot.transpose() * baseline,
                .R = worldToCameraRot,
                .baseline = baseline
            };

            assert(!pose.R.hasNaN());
            for (int j = 0; j < 4; j++) {
                pose.dR[j] = imuToCameraRot * d_worldToCameraRot[j];
                assert(!pose.dR[j].hasNaN());
            }
            trail.push_back(pose);
        }
    }
}

Triangulator::Triangulator(const ParametersOdometry &parameters) :
    parameters(parameters)
{
    const unsigned maxNumPoses = 2 * parameters.cameraTrailLength + 2;
    assert(maxNumPoses >= 2);

    // preallocate matrices
    const unsigned maxDim = maxNumPoses * POSE_DIM;
    dEerror = Eigen::MatrixXd::Zero(3, maxDim + 1);
    dETE = Eigen::MatrixXd::Zero(3, (maxDim + 1) * 3);
}

// Compute triangulated 3d position of a feature given track in image coordinates and
// associated camera poses.
// False return value indicates a singular result (that will likely cause issues later in the algorithm pipeline).
TriangulatorStatus Triangulator::triangulate(
    const TriangulationArgsIn &args,
    TriangulationArgsOut &out,
    odometry::DebugAPI *odometryDebugAPI
) {
    using Eigen::Vector2d;
    using Eigen::Vector3d;
    using Eigen::VectorXd;
    using Eigen::Matrix3d;
    using Eigen::MatrixXd;
    using Eigen::Matrix;

    const vecVector2d &imageFeatures = args.imageFeatures;
    const vecVector2d &featureVelocities = args.featureVelocities;
    const CameraPoseTrail &trail = args.trail;
    Eigen::Vector3d &pf = out.pf;

    if (args.stereo && parameters.useIndependentStereoTriangulation)
        return triangulateStereo(args, out, odometryDebugAPI);

    bool debug = odometryDebugAPI && odometryDebugAPI->publisher;
    const size_t poseCount = trail.size();
    assert(imageFeatures.size() == trail.size());
    assert(poseCount >= 2);
    if (args.estimateImuCameraTimeShift) assert(imageFeatures.size() == featureVelocities.size());

    if (parameters.useLinearTriangulation) {
        TriangulatorStatus status = triangulateLinear(args, out);
        if (debug) {
            odometryDebugAPI->publisher->pushTriangulationPoint(pf.cast<float>());
        }
        return status;
    }

    // Use first and last pose for initial triangulation because they are likely
    // to most differ from each other.
    const size_t ind0 = 0;
    const size_t ind1 = (args.stereo ? poseCount / 2 - 1 : poseCount - 1);
    Matrix<double, 3, 2 * POSE_DIM + 1> dpfTwoCameras;
    const TwoCameraTriangulationArgsIn twoArgs {
        .pose0 = trail.at(ind0),
        .pose1 = trail.at(ind1),
        .ip0 = imageFeatures.at(ind0),
        .ip1 = imageFeatures.at(ind1),
        .velocity0 = &featureVelocities.at(ind0),
        .velocity1 = &featureVelocities.at(ind1),
        .calculateDerivatives = args.calculateDerivatives,
        .estimateImuCameraTimeShift = args.estimateImuCameraTimeShift,
        .derivativeTest = args.derivativeTest,
        .imuToCameraTimeShift = args.imuToCameraTimeShift,
    };
    pf = triangulateWithTwoCameras(twoArgs, &dpfTwoCameras);
    assert(!pf.hasNaN());
    // `pf` and rest of this function take place in the coordinates of `ind0`.

    // Inverse depth parametrisation.
    Matrix3d dpfi_dpf;
    Vector3d pfi = inverseDepth(pf, dpfi_dpf);
    assert(!pfi.hasNaN()); // Can pf(2) == 0?

    const Matrix3d R0T = trail[0].R.transpose();
    if (debug) {
        odometryDebugAPI->publisher->pushTriangulationPoint(inverseToGlobal(pfi, trail[0].p, R0T));
    }

    const size_t dDim = args.calculateDerivatives ? poseCount * POSE_DIM : 0;
    const size_t secondPoseIdx = (args.stereo ? poseCount / 2 - 1 : poseCount - 1);
    // TODO: for stereo mode: to check if it is beneficial to use the last frame from the second camera instead of the reference camera
    //  to do it - you need to alter derivatives inference in triangulateWithTwoCameras
    if (args.calculateDerivatives) {
        out.dpfdp.clear();
        out.dpfdq.clear();
        // The order of derivatives is: p1_x, p1_y, p1_z, q1_w, q1_x, q1_y, q1_z, p2_x, ..., qn_z, t
        dpfi = MatrixXd::Zero(3, dDim + 1);
        dpfi.block<3, POSE_DIM>(0, 0) = dpfTwoCameras.block<3, POSE_DIM>(0, 0);
        dpfi.block<3, POSE_DIM>(0, POSE_DIM * secondPoseIdx) = dpfTwoCameras.block<3, POSE_DIM>(0, POSE_DIM);
        Eigen::Vector3d dpfdt = dpfTwoCameras.block<3, 1>(0, 2 * POSE_DIM);
        dpfi.col(dDim) = dpfi_dpf * dpfdt;

        for (size_t i : { static_cast<size_t>(0), secondPoseIdx }) {
            for (size_t j = 0; j < POSE_DIM; ++j) {
                const size_t colIndex = i * POSE_DIM + j;
                const Vector3d dpf_cur = dpfi.col(colIndex);
                dpfi.col(colIndex) = dpfi_dpf * dpf_cur;
            }
        }
    }

    double rcond = 0.0;
    double Jprev = 1e10;
    bool converged = false;
    const Vector3d &p0 = trail[0].p;
    Eigen::Vector3d dEerrordt;

    for (unsigned optIter = 0; optIter < parameters.triangulationGaussNewtonIterations; optIter++) {
        // Gauss-Newton left-hand-side (ETE) and right-hand-side (Eerror)
        Matrix3d ETE = Matrix3d::Zero();
        Vector3d Eerror = Vector3d::Zero();

        // Derivatives of the above w.r.t. all state variables
        dETE.block(0, 0, 3, (dDim + 1) * 3).setZero();
        dEerror.block(0, 0, 3, dDim + 1).setZero();

        double error2 = 0;

        assert(poseCount >= 2);
        for (size_t i = 0; i < poseCount; ++i) {
            const auto &cur = trail.at(i);

            Matrix3d C = cur.R * R0T;
            Vector3d t = cur.R * (p0 - cur.p);

            Vector3d pfiab(pfi(0), pfi(1), 1.0);
            Vector3d h = C * pfiab + pfi(2) * t;

            const Vector2d h02 = h.segment(0, 2);
            const double ih2sq = 1.0 /(h(2)*h(2));

            Vector2d errorBlock = imageFeatures[i] - h02 / h(2);
            if (args.derivativeTest && args.estimateImuCameraTimeShift) {
                errorBlock += args.imuToCameraTimeShift * featureVelocities[i];
            }
            assert(!errorBlock.hasNaN());

            // Jacobian of error w.r.t. pfi
            Matrix23 Eblock;
            Eblock.topLeftCorner<2, 2>() = (-1/h(2)) * C.block(0, 0, 2, 2) + h02 * ih2sq * C.block(2, 0, 1, 2);
            Eblock.block<2, 1>(0, 2) = -t.segment(0, 2) / h(2) + h02 * ih2sq * t(2);

            error2 += errorBlock.squaredNorm();

            ETE += Eblock.transpose() * Eblock;
            Eerror += Eblock.transpose() * errorBlock;

            if (args.calculateDerivatives && args.estimateImuCameraTimeShift) {
                const Vector3d dpfiab(dpfi(0, dDim), dpfi(1, dDim), 0);
                const Vector3d dh = C * dpfiab + dpfi(2, dDim) * t;
                const double dih2 = -dh(2) / (h(2)*h(2));
                const Vector2d dh02 = dh.segment<2>(0);
                const double dih2sq = -2 * dh(2) * ih2sq / h(2);
                const Vector2d dErrorBlock = featureVelocities[i] - dh02 / h(2) - dih2 * h02;
                Matrix23 dEblock;
                dEblock.topLeftCorner<2, 2>() =
                    -dih2 * C.block(0, 0, 2, 2) + (dh02 * ih2sq + dih2sq * h02) * C.block(2, 0, 1, 2);
                dEblock.block<2, 1>(0, 2) =
                    -t.segment(0, 2) * dih2 + dh02 * ih2sq * t(2) + h02 * dih2sq * t(2);
                dEerror.col(dDim) += dEblock.transpose() * errorBlock + Eblock.transpose() * dErrorBlock;
                dETE.block<3, 3>(0, dDim * 3) += dEblock.transpose() * Eblock + Eblock.transpose() * dEblock;
            }
            // skipped if !calculateDerivatives
            for (size_t j = 0; j < dDim; ++j) {
                const size_t poseIdx = j / POSE_DIM;
                const size_t componentIdx = j % POSE_DIM;
                const bool currentPose = poseIdx == i;

                Matrix3d dRi = Matrix3d::Zero();
                Matrix3d dR0 = Matrix3d::Zero();
                Vector3d dp0 = Vector3d::Zero();
                Vector3d dpi = Vector3d::Zero();

                if (componentIdx < 3) {
                    if (currentPose) dpi(componentIdx) = 1;
                    if (poseIdx == 0) dp0(componentIdx) = 1;
                } else {
                    const size_t quaternionIdx = componentIdx - 3;
                    assert(quaternionIdx < 4);
                    if (currentPose) {
                        const auto &pose = trail.at(poseIdx);
                        dRi = pose.dR[quaternionIdx];
                        dpi = -dRi.transpose() * pose.baseline;
                    }
                    if (poseIdx == 0) {
                        const auto &pose = trail.at(poseIdx);
                        dR0 = pose.dR[quaternionIdx];
                        dp0 = -dR0.transpose() * pose.baseline;
                    }
                }

                const Matrix3d dC = dRi * R0T + cur.R * dR0.transpose();
                const Vector3d dt = dRi * (p0 - cur.p) + cur.R * (dp0 - dpi);
                const Vector3d dpfiab(dpfi(0, j), dpfi(1, j), 0);
                const Vector3d dh = dC * pfiab + C * dpfiab + dpfi(2, j) * t + pfi(2) * dt;
                const double dih2 = -dh(2) / (h(2)*h(2));

                const Vector2d dh02 = dh.segment<2>(0);
                const double dih2sq = -2 * dh(2) * ih2sq / h(2);

                const Vector2d dErrorBlock = -dh02 / h(2) - dih2 * h02;

                Matrix23 dEblock;
                dEblock.topLeftCorner<2, 2>() =
                    -dih2 * C.block(0, 0, 2, 2) + (-1/h(2)) * dC.block(0, 0, 2, 2) +
                    (dh02 * ih2sq + dih2sq * h02) * C.block(2, 0, 1, 2) +
                    h02 * ih2sq * dC.block(2, 0, 1, 2);
                dEblock.block<2, 1>(0, 2) =
                    -dt.segment(0, 2) / h(2) -
                    t.segment(0, 2) * dih2 +
                    dh02 * ih2sq * t(2) +
                    h02 * dih2sq * t(2) +
                    h02 * ih2sq * dt(2);

                dEerror.col(j) += dEblock.transpose() * errorBlock + Eblock.transpose() * dErrorBlock;
                dETE.block<3, 3>(0, j * 3) += dEblock.transpose() * Eblock + Eblock.transpose() * dEblock;
            }
        }

        // Matlab: theta += -(E'*E) \ (E'*error);
        const Eigen::LDLT<Eigen::Matrix3d> X = ETE.ldlt();
        pfi += -X.solve(Eerror);
        assert(!pfi.hasNaN());
        if (debug) {
            odometryDebugAPI->publisher->pushTriangulationPoint(inverseToGlobal(pfi, p0, R0T));
        }

        // skipped if !calculateDerivatives
        for (size_t j = 0; j < dDim + 1; ++j) {
            // the derivative of a matrix inverse is d(A^-1) = -A^-1 dA A^-1
            const Vector3d dX_Eerror = -X.solve(dETE.block<3, 3>(0, j * 3) * X.solve(Eerror));
            dpfi.col(j) += -X.solve(dEerror.col(j)) - dX_Eerror;
        }
        assert(!dpfi.hasNaN());

        rcond = X.rcond();

        // Check convergence.
        double Rnoise = pow2(parameters.triangulationConvergenceR);
        double J = 0.5 * error2 / Rnoise;
        double Jd = std::abs((J - Jprev) / J);
        Jprev = J;

        if (Jd < parameters.triangulationConvergenceThreshold) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        // Usually happens when the camera trail has jumps and plausible
        // triangulation cannot be solved.
        return TriangulatorStatus::NO_CONVERGENCE;
    }
    if (rcond < parameters.triangulationRcondThreshold) {
        // Usually happens when there's little camera movement and the 2D points are
        // all very close to each other.
        return TriangulatorStatus::BAD_COND;
    }

    // Revert to normal coordinates.
    Matrix3d dpf0_dpfi;
    const Vector3d pf0 = inverseDepth(pfi, dpf0_dpfi);
    pf = R0T * pf0 + p0;
    assert(!pf.hasNaN());

    // Failure where pfi(2) becomes very large. Should investigate what causes it. Same as bad rcond?
    if (pf == p0) {
        return TriangulatorStatus::UNKNOWN_PROBLEM;
    }

    // TODO Comparing the value of `J` (scaled by number of track points) to a threshold
    // could be another useful test. Matlab source might do it.

    // convert "dpfi" (inverse depth) back to "dpf" (xyz)
    // skipped if !calculateDerivatives
    for (size_t j = 0; j < dDim + 1; ++j) {
        Vector3d dp0 = Vector3d::Zero();
        if (j < 3) dp0(j) = 1;
        Matrix3d dR0T = Matrix3d::Zero();
        if (j >= 3 && j < POSE_DIM) dR0T = trail.at(0).dR[j - 3].transpose();
        dpfi.col(j) = dR0T * pf0 + R0T * dpf0_dpfi * dpfi.col(j) + dp0;
    }
    if (args.calculateDerivatives) {
        for (size_t j = 0; j < poseCount; ++j) {
            out.dpfdp.push_back(dpfi.block<3, 3>(0, j * POSE_DIM));
            out.dpfdq.push_back(dpfi.block<3, 4>(0, j * POSE_DIM + 3));
        }
        out.dpfdt = dpfi.col(dpfi.cols() - 1);
    }

    // Check if the triangulated point lands behind any of the cameras.
    if (isBehind(pf, trail)) return TriangulatorStatus::BEHIND;

    // Reject triangulations that land far.
    // if ((pf - p[0]).norm() > 100) {
    //     return TriangulatorStatus::BAD_COND;
    // }

    return TriangulatorStatus::OK;
}

TriangulatorStatus Triangulator::triangulateStereo(
    const TriangulationArgsIn &args,
    TriangulationArgsOut &out,
    odometry::DebugAPI *odometryDebugAPI)
{
    using Eigen::Vector2d;
    using Eigen::Vector3d;
    using Eigen::Matrix3d;
    const bool debug = odometryDebugAPI && odometryDebugAPI->publisher;
    const auto &trail = args.trail;

    Vector3d weightedSum = Vector3d::Zero();
    Matrix3d sumOfWeights = Matrix3d::Zero();

    dWeightedSum = Eigen::MatrixXd::Zero(3, trail.size() * POSE_DIM + 1);
    dSumOfWeights = Eigen::MatrixXd::Zero(3, dWeightedSum.cols() * 3);

    const Vector3d p0 = trail[0].p;
    const Matrix3d R0 = trail[0].R;
    constexpr double MIN_NORM_TO_INVERT = 1e-10;

    for (std::size_t i = 0; i < trail.size(); ++i) {
        const auto &pose = trail.at(i);
        if (!pose.hasFeature3D) continue;

        const Matrix3d RT = pose.R.transpose();

        Matrix3d dfeature3D;
        const Vector3d feature3D = inverseDepth(pose.feature3DIdp, dfeature3D);

        const Vector3d pos = RT * feature3D + pose.p;
        assert(!pos.hasNaN());
        if (debug) {
            odometryDebugAPI->publisher->pushTriangulationPoint(pos.cast<float>());
        }

        Matrix3d dipos, ddipos[3];
        const Vector3d pos0 = R0 * (pos - p0);
        const Vector3d ipos = inverseDepth(pos0, dipos, ddipos);

        const Matrix3d covJ = dipos * R0 * RT * dfeature3D;
        const Matrix3d cov = covJ * pose.feature3DCov * covJ.transpose();

        if (cov.norm() < MIN_NORM_TO_INVERT) continue;

        const Matrix3d info = cov.inverse();
        assert(!info.hasNaN());

        weightedSum += info * ipos;
        sumOfWeights += info;

        const int offs = i * POSE_DIM;
        for (int j = 0; j < 3; ++j) {
            Vector3d dpos = Vector3d::Zero();
            dpos(j) = 1;
            const Vector3d dpos0 = R0 * dpos;

            const Matrix3d ddipos_dj = sumSecondDerivative(ddipos, dpos0);
            const Matrix3d dcovJ = ddipos_dj * R0 * RT * dfeature3D;
            const Matrix3d dcov = dcovJ * pose.feature3DCov * covJ.transpose() + covJ * pose.feature3DCov * dcovJ.transpose();
            const Matrix3d dinfo = -info * dcov * info;

            const Vector3d dWs = info * dipos * dpos0 + dinfo * ipos;
            dWeightedSum.col(offs + j) = dWs;
            dSumOfWeights.block<3, 3>(0, (offs + j) * 3) = dinfo;

            // first pose is special
            const Vector3d dWs0 = -dWs; // minus sign due to -p0
            const Matrix3d dinfo0 = -dinfo;
            dWeightedSum.col(0 + j) += dWs0;
            dSumOfWeights.block<3, 3>(0, (0 + j) * 3) += dinfo0;
        }
        for (int j = 0; j < 4; ++j) {
            const Matrix3d dR = pose.dR[j];
            const Matrix3d dRT = dR.transpose();
            const Vector3d dpos = dRT * feature3D;
            const Vector3d dpos0 = R0 * dpos;
            const Vector3d dipos_dj = dipos * dpos0;
            Matrix3d ddipos_dj = sumSecondDerivative(ddipos, dpos0);
            const Matrix3d dcovJ = ddipos_dj * R0 * RT * dfeature3D + dipos * R0 * dRT * dfeature3D;

            const Matrix3d dcov = dcovJ * pose.feature3DCov * covJ.transpose() + covJ * pose.feature3DCov * dcovJ.transpose();
            const Matrix3d dinfo = -info * dcov * info;

            dWeightedSum.col(offs + j + 3) = info * dipos_dj + dinfo * ipos;
            dSumOfWeights.block<3, 3>(0, (offs + j + 3) * 3) = dinfo;

            // first pose is special
            const Matrix3d dR0 = trail.at(0).dR[j];
            const Vector3d dpos00 = dR0 * (pos - p0);
            const Vector3d dipos_dj0 = dipos * dpos00;
            Matrix3d ddipos_dj0 = sumSecondDerivative(ddipos, dpos00);

            const Matrix3d dcovJ0 = ddipos_dj0 * R0 * RT * dfeature3D + dipos * dR0 * RT * dfeature3D;
            const Matrix3d dcov0 = dcovJ0 * pose.feature3DCov * covJ.transpose() + covJ * pose.feature3DCov * dcovJ0.transpose();
            const Matrix3d dinfo0 = -info * dcov0 * info;

            dWeightedSum.col(0 + j + 3) += info * dipos_dj0 + dinfo0 * ipos;
            dSumOfWeights.block<3, 3>(0, (0 + j + 3) * 3) += dinfo0;
        }
        if (parameters.estimateImuCameraTimeShift) {
            const size_t t_idx = dWeightedSum.cols() - 1;

            {
                const Vector2d velocity = args.featureVelocities[i];
                const Vector3d dFeature3dIdp_dt = Vector3d(velocity(0), velocity(1), 0.0);
                const Vector3d dFeature3d_dt = dfeature3D * dFeature3dIdp_dt;
                const Vector3d dpos = RT * dFeature3d_dt;
                const Vector3d dpos0 = R0 * dpos;

                const Vector3d dipos_dj = dipos * dpos0;
                Matrix3d ddipos_dj = sumSecondDerivative(ddipos, dpos0);
                const Matrix3d dcovJ = ddipos_dj * R0 * RT * dfeature3D;

                const Matrix3d dcov = dcovJ * pose.feature3DCov * covJ.transpose() + covJ * pose.feature3DCov * dcovJ.transpose();
                const Matrix3d dinfo = -info * dcov * info;

                dWeightedSum.col(t_idx) += info * dipos_dj + dinfo * ipos;
                dSumOfWeights.block<3, 3>(0, t_idx * 3) += dinfo;
            }
            {
                // first pose is special
                const Vector2d velocity0 = args.featureVelocities[0];
                const Vector3d dFeature3dIdp_dt = Vector3d(velocity0(0), velocity0(1), 0.0);
                const Vector3d dFeature3d_dt = dfeature3D * dFeature3dIdp_dt;
                const Vector3d dpos00 = -dFeature3d_dt; // minus sign due to -p0 and R0/R0T cancel out

                const Vector3d dipos_dj0 = dipos * dpos00;
                Matrix3d ddipos_dj0 = sumSecondDerivative(ddipos, dpos00);
                const Matrix3d dcovJ0 = ddipos_dj0 * dfeature3D;
                const Matrix3d dcov0 = dcovJ0 * pose.feature3DCov * covJ.transpose() + covJ * pose.feature3DCov * dcovJ0.transpose();
                const Matrix3d dinfo0 = -info * dcov0 * info;

                dWeightedSum.col(t_idx) += info * dipos_dj0 + dinfo0 * ipos;
                dSumOfWeights.block<3, 3>(0, t_idx * 3) += dinfo0;
            }
        }
    }

    if (sumOfWeights.norm() < MIN_NORM_TO_INVERT) return TriangulatorStatus::BAD_COND;

    const Matrix3d invSumOfWeights = sumOfWeights.inverse();
    assert(!invSumOfWeights.hasNaN());

    const Vector3d pfi = invSumOfWeights * weightedSum;

    Matrix3d dpf_dpfi;
    const Vector3d pf0 = inverseDepth(pfi, dpf_dpfi);
    out.pf = R0.transpose() * pf0 + p0;
    dpf_dpfi = R0.transpose() * dpf_dpfi;

    assert(!out.pf.hasNaN());

    if (isBehind(out.pf, trail)) return TriangulatorStatus::BEHIND;

    out.dpfdp.clear();
    out.dpfdq.clear();

    for (std::size_t i = 0; i < trail.size(); ++i) {
        Eigen::Matrix3d dpfdp;
        Eigen::Matrix<double, 3, 4> dpfdq;
        for (unsigned j = 0; j < POSE_DIM; ++j) {
            const int idx = i * POSE_DIM + j;
            const Matrix3d dSumW = dSumOfWeights.block<3, 3>(0, idx * 3);
            const Vector3d dW = dWeightedSum.col(idx);
            const Matrix3d dInvSumW = -invSumOfWeights * dSumW * invSumOfWeights;
            Eigen::Vector3d dfpcur = dpf_dpfi * (dInvSumW * weightedSum + invSumOfWeights * dW);
            if (i == 0) {
                if (j < 3) {
                    Vector3d dp0 = Vector3d::Zero();
                    dp0(j) = 1;
                    dfpcur += dp0;
                } else {
                    const Matrix3d dR0T = trail.at(0).dR[j - 3].transpose();
                    dfpcur += dR0T * pf0;
                }
            }
            if (j < 3) {
                dpfdp.col(j) = dfpcur;
            } else {
                dpfdq.col(j - 3) = dfpcur;
            }
        }
        assert(!dpfdp.hasNaN());
        assert(!dpfdq.hasNaN());
        out.dpfdp.push_back(dpfdp);
        out.dpfdq.push_back(dpfdq);
    }
    {
        // time shift param
        const int idx = trail.size();
        const Matrix3d dSumW = dSumOfWeights.block<3, 3>(0, idx * 3);
        const Vector3d dW = dWeightedSum.col(idx);
        const Matrix3d dInvSumW = -invSumOfWeights * dSumW * invSumOfWeights;
        out.dpfdt = dpf_dpfi * (dInvSumW * weightedSum + invSumOfWeights * dW);
        assert(!out.dpfdt.hasNaN());
    }

    return TriangulatorStatus::OK;
}

Eigen::Vector3d triangulateWithTwoCameras(
    const TwoCameraTriangulationArgsIn &args,
    Eigen::Matrix<double, 3, 2 * POSE_DIM + 1> *dpf
) {
    using Eigen::Matrix;
    using Eigen::Matrix2d;
    using Eigen::Matrix3d;
    using Eigen::Vector2d;
    using Eigen::Vector3d;

    const Matrix3d &R0 = args.pose0.R;
    const Matrix3d &R1 = args.pose1.R;
    const Vector3d &p0 = args.pose0.p;
    const Vector3d &p1 = args.pose1.p;
    Matrix3d C = R0 * R1.transpose();
    Vector3d b = R0 * (p1 - p0);

    Vector3d v0(args.ip0(0), args.ip0(1), 1.0);
    Vector3d v1(args.ip1(0), args.ip1(1), 1.0);
    if (args.derivativeTest && args.estimateImuCameraTimeShift) {
        v0.segment<2>(0) += args.imuToCameraTimeShift * *args.velocity0;
        v1.segment<2>(0) += args.imuToCameraTimeShift * *args.velocity1;
    }
    Vector3d vn0 = v0;
    Vector3d vn1 = v1;
    vn0.normalize();
    vn1.normalize();

    Matrix32 A;
    A.col(0) = vn0;
    A.col(1) = -C * vn1;

    const Matrix23 iA = pinv(A);
    const Vector2d s = iA * b;
    const Vector3d pf = s(0) * vn0;

    if (!args.calculateDerivatives) return pf;

    assert(dpf);
    constexpr unsigned IDX_P0 = 0;
    constexpr unsigned IDX_Q0 = 3;
    constexpr unsigned IDX_P1 = 7;
    constexpr unsigned IDX_Q1 = 10;
    constexpr unsigned DIM_D = 14;
    constexpr unsigned IDX_T = DIM_D;
    constexpr unsigned DIM_ALL = DIM_D + 1;

    Matrix32 dA[DIM_ALL];
    Vector3d db[DIM_ALL];
    for (unsigned i = 0; i < DIM_ALL; ++i) {
        dA[i].setZero();
        db[i].setZero();
    }

    for (int i = 0; i < 4; ++i) {
        const Matrix3d dC_dq0 = args.pose0.dR[i] * R1.transpose();
        const Matrix3d dC_dq1 = R0 * args.pose1.dR[i].transpose();

        dA[IDX_Q0 + i].col(1) = -dC_dq0 * vn1;
        dA[IDX_Q1 + i].col(1) = -dC_dq1 * vn1;
        db[IDX_Q0 + i] = args.pose0.dR[i] * (p1 - p0);

        if (i < 3) {
            db[IDX_P0 + i] = -R0.col(i);
            db[IDX_P1 + i] = R0.col(i);
        }
    }

    for (unsigned i = 0; i < DIM_D; ++i) {
        const Matrix23 diA = dpinv(A, iA, dA[i]);
        // MATLAB: dscalar = (iA*db + diA*b);
        const auto dscalar = (iA * db[i] + diA * b);
        // MATLAB: dp_f1_1(:,i) = dscalar(1)*v_1;
        dpf->block<3, 1>(0, i) = dscalar(0) * vn0;
    }

    if (args.estimateImuCameraTimeShift) {
        Eigen::Matrix3d B0 = Eigen::Matrix3d::Identity() - vn0 * vn0.transpose();
        Eigen::Matrix3d B1 = Eigen::Matrix3d::Identity() - vn1 * vn1.transpose();
        // Derivative of normalized v wrt v.
        double n0 = v0.norm();
        double n1 = v1.norm();
        Eigen::Matrix3d dvn0dv0 = B0 / n0;
        Eigen::Matrix3d dvn1dv1 = B1 / n1;
        assert(args.velocity0 && args.velocity1);
        Vector2d velocity0_ = *args.velocity0;
        Vector2d velocity1_ = *args.velocity1;
        dA[IDX_T].col(0) = dvn0dv0 * Vector3d(velocity0_(0), velocity0_(1), 0.0);
        dA[IDX_T].col(1) = -C * dvn1dv1 * Vector3d(velocity1_(0), velocity1_(1), 0.0);
        Vector3d dvn0dt = dA[IDX_T].col(0);
        const Matrix23 diA = dpinv(A, iA, dA[IDX_T]);
        double ds0dt = diA.row(0) * b;
        dpf->block<3, 1>(0, IDX_T) = s(0) * dvn0dt + vn0 * ds0dt;
    }
    else {
        dpf->block<3, 1>(0, IDX_T) = Eigen::Vector3d::Zero();
    }

    return pf;
}

bool triangulateStereoFeatureIdp(
    const Eigen::Vector2d &ipFirst,
    const Eigen::Vector2d &ipSecond,
    const Eigen::Matrix4d &secondToFirstCamera,
    Eigen::Vector3d &triangulatedPointIdp,
    Eigen::Matrix3d *triangulatedCov)
{
    using Eigen::Vector2d;
    using Eigen::Vector3d;
    using Eigen::Matrix3d;

    // Implements this triangulation method: (w)Mid2
    // https://bmvc2019.org/wp-content/uploads/papers/0331-paper.pdf

    // Here the formulas are interpreted so that 0 in the paper means second
    // camera and 1 means the first camera, that is, the meaning of 0 and 1
    // are flipped compared to triangulateStereoLinear & mid-point
    const Vector3d f0(ipSecond(0), ipSecond(1), 1.0);
    const Vector3d f1(ipFirst(0), ipFirst(1), 1.0);

    const Vector3d f0hat = f0.normalized();
    const Vector3d f1hat = f1.normalized();

    const Matrix3d R = secondToFirstCamera.topLeftCorner<3, 3>();
    const Vector3d t = secondToFirstCamera.block<3, 1>(0, 3);

    const Vector3d p = (R * f0hat).cross(f1hat);
    const Vector3d q = (R * f0hat).cross(t);
    const Vector3d r = f1hat.cross(t);
    const double pn = p.norm();
    const double qn = q.norm();
    const double rn = r.norm();

    const double lambda0 = rn / pn;
    const double lambda1 = qn / pn;
    const double w = qn / (qn + rn);
    const Vector3d pf = w * (t + lambda0 * (R * f0hat + f1hat));

    const Vector3d l0Rf0hat = lambda0 * R * f0hat;
    const Vector3d l1f1hat = lambda1 * f1hat;

    const double
        c0 = (t + l0Rf0hat - l1f1hat).squaredNorm(),
        c1 = (t + l0Rf0hat + l1f1hat).squaredNorm(),
        c2 = (t - l0Rf0hat - l1f1hat).squaredNorm(),
        c3 = (t - l0Rf0hat + l1f1hat).squaredNorm();

    if (c0 > std::min(std::min(c1, c2), c3)) {
        return false;
    }
    else if (!triangulatedCov) {
        triangulatedPointIdp = Eigen::Vector3d(pf.x(), pf.y(), 1) / pf.z();
        return true;
    }

    // sensitivity
    Matrix3d dpf0hat, dpf1hat;
    for (int i = 0; i < 3; ++i) {
        // d?i = df?hat_i
        Vector3d d0i = Vector3d::Zero(), d1i = Vector3d::Zero();
        d0i(i) = 1;
        d1i(i) = 1;

        const Vector3d dp_d0i = (R * d0i).cross(f1hat);
        const Vector3d dp_d1i = (R * f0hat).cross(d1i);
        const Vector3d dq_d0i = (R * d0i).cross(t);
        const Vector3d dr_d1i = d1i.cross(t);

        const double dpn_d0i = dp_d0i.dot(p) / pn;
        const double dpn_d1i = dp_d1i.dot(p) / pn;
        const double dqn_d0i = dq_d0i.dot(q) / qn;
        const double drn_d1i = dr_d1i.dot(r) / qn;

        const double dlambda0_d0i = -lambda0 / pn * dpn_d0i;
        const double dlambda0_d1i = -lambda0 / pn * dpn_d1i + drn_d1i / pn;

        // w = (qn / (qn + rn))
        const double dw_d0i = dqn_d0i / (qn + rn) - w / (qn + rn) * dqn_d0i;
        const double dw_d1i = - w / (qn + rn) * drn_d1i;

        const Vector3d vInner = R * f0hat + f1hat;
        dpf0hat.col(i) =
            dw_d0i * (t + lambda0 * vInner) +
            w * dlambda0_d0i * vInner +
            w * lambda0 * R * d0i;
        dpf1hat.col(i) =
            dw_d1i * (t + lambda0 * vInner) +
            w * dlambda0_d1i * vInner +
            w * lambda0 * d1i;
    }

    const Matrix3d df0 = (Matrix3d::Identity() - f0hat*f0hat.transpose())  / f0.norm();
    const Matrix3d df1 = (Matrix3d::Identity() - f1hat*f1hat.transpose())  / f1.norm();

    const Matrix32 dpf_df0 = dpf0hat * df0.topLeftCorner<3, 2>();
    const Matrix32 dpf_df1 = dpf1hat * df1.topLeftCorner<3, 2>();

    // Inverse depth parametrization
    Matrix3d dpfi_dpf;
    triangulatedPointIdp = inverseDepth(pf, dpfi_dpf);
    const Matrix32 dpfi_df0 = dpfi_dpf * dpf_df0;
    const Matrix32 dpfi_df1 = dpfi_dpf * dpf_df1;

    // assuming isotropic error in normalized pixel coordinates (TODO: this can be improved)
    *triangulatedCov = dpfi_df0 * dpfi_df0.transpose() + dpfi_df1 * dpfi_df1.transpose();

    return true;
}

TriangulatorStatus triangulateLinear(
    const TriangulationArgsIn &args,
    TriangulationArgsOut &out
) {
    const vecVector2d &imageFeatures = args.imageFeatures;
    const vecVector2d &featureVelocities = args.featureVelocities;
    const CameraPoseTrail &trail = args.trail;

    const int poseCount = static_cast<int>(trail.size());
    Eigen::Matrix3d S0 = Eigen::Matrix3d::Zero();
    Eigen::Vector3d S1 = Eigen::Vector3d::Zero();
    for (int i = 0; i < poseCount; ++i) {
        const Eigen::Vector2d &ip = args.derivativeTest
            ? imageFeatures[i] + args.imuToCameraTimeShift * featureVelocities[i]
            : imageFeatures[i];
        // v is direction of the camera ray, vn after normalization.
        const auto &pose = trail.at(i);
        Eigen::Vector3d vn = pose.R.transpose() * Eigen::Vector3d(ip(0), ip(1), 1.0);
        vn.normalize();
        Eigen::Matrix3d A = Eigen::Matrix3d::Identity() - vn * vn.transpose();
        S0 += A;
        S1 += A * pose.p;
    }
    Eigen::Matrix3d S0inv = S0.inverse();
    out.pf = S0inv * S1;

    if (args.calculateDerivatives) {
        out.dpfdp.clear();
        out.dpfdq.clear();
        out.dpfdt = Eigen::Vector3d::Zero();
        for (int i = 0; i < poseCount; ++i) {
            const Eigen::Vector2d &ip = args.derivativeTest
                ? imageFeatures[i] + args.imuToCameraTimeShift * featureVelocities[i]
                : imageFeatures[i];
            const auto &pose = trail.at(i);
            Eigen::Vector3d v = pose.R.transpose() * Eigen::Vector3d(ip(0), ip(1), 1.0);
            Eigen::Vector3d vn = v;
            vn.normalize();
            Eigen::Matrix3d A = Eigen::Matrix3d::Identity() - vn * vn.transpose();
            out.dpfdp.push_back(S0inv * A);

            // Derivative of v wrt q.
            Eigen::Matrix<double, 3, 4> dvdq = Eigen::Matrix<double, 3, 4>::Zero();
            for (int k = 0; k < 4; ++k) {
                dvdq.block<3, 1>(0, k) = pose.dR[k].transpose() * Eigen::Vector3d(ip(0), ip(1), 1.0);
            }

            // Derivative of normalized v wrt v.
            double n = v.norm();
            Eigen::Matrix3d dvndv = A / n;

            // Derivative of p wrt normalized v.
            Eigen::Matrix3d dpfdvn = Eigen::Matrix3d::Zero();
            for (int k = 0; k < 3; ++k) {
                // Derivatives of v*v' wrt to v. Since first is 3x3 and second 3x1,
                // differentiate wrt to individual components of v.
                Eigen::Vector3d ek = Eigen::Vector3d::Zero();
                ek(k) = 1;
                Eigen::Matrix3d Q = ek * vn.transpose() + vn * ek.transpose();
                dpfdvn.block<3, 1>(0, k) = S0inv * Q * S0inv * S1 - S0inv * Q * pose.p;
            }

            out.dpfdq.push_back(dpfdvn * dvndv * dvdq);

            if (args.estimateImuCameraTimeShift) {
                Eigen::Vector3d dvdt = pose.R.transpose()
                    * Eigen::Vector3d(featureVelocities[i](0), featureVelocities[i](1), 0);
                out.dpfdt += dpfdvn * dvndv * dvdt;
            }
        }
    }

    // Check the point is in front of all the cameras.
    if (isBehind(out.pf, trail)) return TriangulatorStatus::BEHIND;
    return TriangulatorStatus::OK;
}

PrepareVuStatus prepareVisualUpdate(
    const PrepareVisualUpdateArgsIn &args,
    Eigen::MatrixXd &H,
    Eigen::VectorXd &y
) {
    const vecVector2d &featureVelocities = args.featureVelocities;
    const CameraPoseTrail &trail = args.trail;
    const std::vector<int> &poseTrailIndex = args.poseTrailIndex;
    const int nValid = static_cast<int>(trail.size());
    assert(nValid > 0);

    int endIdx = 0;
    if (args.truncated) {
        // truncate H matrix to save CPU cycles with short tracks
        for (int idx : poseTrailIndex) {
            int jPos, jOri;
            getPosOriIndices(idx, jPos, jOri);
            endIdx = std::max(endIdx, std::max(jPos + 3, jOri + 4));
        }
        if (args.mapPointOffset > 0) {
            endIdx = args.mapPointOffset + 3;
        }
    } else {
        endIdx = args.stateDim;
    }

    H = Eigen::MatrixXd::Zero(2 * nValid, endIdx);
    y = Eigen::VectorXd::Zero(2 * nValid);

    for (std::size_t i = 0; i < trail.size(); ++i) {
        std::size_t trailIndex = i % poseTrailIndex.size();
        const auto &pose = trail.at(i);

        assert(!pose.p.hasNaN());
        assert(!pose.R.hasNaN());
        Eigen::Vector3d pt = args.triangulationOut.pf - pose.p;
        // Triangulated 3D point in camera coordiantes of pose i
        Eigen::Vector3d pfc = pose.R * pt;

        // Check if the point is behind camera.
        if (pfc(2) == 0) {
            log_debug("Point depth 0.");
            // This is fatal for processing this track.
            return PREPARE_VU_ZERO_DEPTH;
        } else if (pfc(2) < 0) {
            return PREPARE_VU_BEHIND;
        }

        // project to normalized pixel coordinates
        // and compute the Jacobian of that projection
        Eigen::Matrix3d dipHomog;
        const Eigen::Vector3d ipHomog = inverseDepth(pfc, dipHomog);

        y.segment<2>(2 * i) = ipHomog.segment<2>(0); // "ip"
        if (args.derivativeTest && args.estimateImuCameraTimeShift) {
            y.segment<2>(2 * i) -= args.imuToCameraTimeShift * featureVelocities[i];
        }
        const Matrix23 dip = dipHomog.topLeftCorner<2, 3>();

        int iPos, iOri;
        getPosOriIndices(poseTrailIndex[trailIndex], iPos, iOri);

        // H is the Jacobian of the non-linear measurement function:
        //   y[i] = h(m, pfc[i]) = projectToCamera(pose.R * (pf - pose.p))
        Eigen::Matrix<double, 3, 4> dRpt;
        for (int j = 0; j < 4; j++) {
            dRpt.col(j) = pose.dR[j] * pt + pose.R * pose.dR[j].transpose() * pose.baseline;
        }
        H.block<2, 3>(2 * i, iPos) = -dip * pose.R;
        H.block<2, 4>(2 * i, iOri) = dip * dRpt;

        if (args.triangulationOut.dpfdp.size() > 0) {
            const auto &o = args.triangulationOut;
            assert(o.dpfdp.size() == poseTrailIndex.size());
            assert(o.dpfdq.size() == poseTrailIndex.size());
            for (size_t j = 0; j < poseTrailIndex.size(); j++) {
                int jPos, jOri;
                getPosOriIndices(poseTrailIndex[j], jPos, jOri);
                H.block<2, 3>(2 * i, jPos) += dip * pose.R * o.dpfdp[j];
                H.block<2, 4>(2 * i, jOri) += dip * pose.R * o.dpfdq[j];
            }
            if (args.estimateImuCameraTimeShift) {
                H.block(2 * i, SFT, 2, 1) = dip * pose.R * o.dpfdt - featureVelocities[i];
            }
        }
        if (args.mapPointOffset > 0) {
            H.block<2, 3>(2* i, args.mapPointOffset) += dip * pose.R;
        }
    }
    return PREPARE_VU_OK;
}

void getPosOriIndices(int i, int& pos, int& ori) {
    if (i == 0) {
        pos = POS;
        ori = ORI;
    }
    else {
        pos = CAM + 7 * (i - 1);
        ori = CAM + 7 * (i - 1) + 3;
    }
}

// Moore-Penrose pseudo inverse.
// May not be a particularly fast way to do this
Matrix23 pinv(const Matrix32& A) {
    return A.completeOrthogonalDecomposition().pseudoInverse();
}

Eigen::Vector3d inverseDepth(const Eigen::Vector3d &p, Eigen::Matrix3d &dip, Eigen::Matrix3d *ddip) {
    // not really "depth" / distance for Fisheye... could be improved?
    const Eigen::Vector3d ip = Eigen::Vector3d(p.x(), p.y(), 1) / p.z();
    dip = Eigen::Matrix3d::Zero();
    dip.topLeftCorner<2, 2>() = Eigen::Matrix2d::Identity() / p.z();
    dip.col(2) = -ip / p.z();

    if (ddip != nullptr) {
        const double ipz2 = 1.0 / (p.z() * p.z());
        for (int i = 0; i < 3; ++i) {
            Eigen::Matrix3d &ddip_i = ddip[i];
            ddip_i.setZero();
            if (i < 2) {
                Eigen::Vector3d dp = Eigen::Vector3d::Zero();
                dp(i) = 1;
                ddip_i.col(2) = -dp * ipz2;
            } else {
                ddip_i.topLeftCorner<2, 2>() = -Eigen::Matrix2d::Identity() * ipz2;
                ddip_i.col(2) = 2 * ip * ipz2;
            }
        }
    }

    return ip;
}

} // namespace odometry
