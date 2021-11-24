#include "output.hpp"
#include "ekf.hpp"
#include "ekf_state_index.hpp"

#include "../util/logging.hpp"

namespace odometry {
Eigen::Vector3d Output::position() const { return inertialMean.segment<3>(POS); }
Eigen::Vector3d Output::velocity() const { return inertialMean.segment<3>(VEL); }
Eigen::Vector4d Output::orientation() const { return inertialMean.segment<4>(ORI); }
Eigen::Matrix3d Output::positionCovariance() const { return positionCov; }
Eigen::Matrix3d Output::velocityCovariance() const { return velocityCov; }
Eigen::Vector3d Output::meanBGA() const { return inertialMean.segment<3>(BGA); }
Eigen::Vector3d Output::meanBAA() const { return inertialMean.segment<3>(BAA); }
Eigen::Vector3d Output::meanBAT() const { return inertialMean.segment<3>(BAT); }
Eigen::Vector3d Output::covDiagBGA() const { return inertialCovDiag.segment<3>(BGA); }
Eigen::Vector3d Output::covDiagBAA() const { return inertialCovDiag.segment<3>(BAA); }
Eigen::Vector3d Output::covDiagBAT() const { return inertialCovDiag.segment<3>(BAT); }

size_t Output::poseTrailOffset(int idx) const {
    return INER_DIM + idx * POSE_DIM;
}

size_t Output::poseTrailLength() const {
    if (!poseTrailTimeStamps) return 0;
    return poseTrailTimeStamps->size();
}

Eigen::Vector3d Output::poseTrailPosition(int idx) const {
    assert(fullMean);
    assert(idx >= 0);
    const int offs = poseTrailOffset(idx);
    assert(offs <= fullMean->rows() - 3);
    return fullMean->segment<3>(offs);
}

Eigen::Vector4d Output::poseTrailOrientation(int idx) const {
    assert(fullMean);
    assert(idx >= 0);
    const int offs = poseTrailOffset(idx) + 3;
    assert(offs <= fullMean->rows() - 4);
    return fullMean->segment<4>(offs);
}

double Output::poseTrailTimeStamp(int idx) const {
    assert(poseTrailTimeStamps);
    return poseTrailTimeStamps->at(idx);
}

void Output::setFromEKF(
    const EKF &ekf,
    const EKFStateIndex &stateIndex,
    std::shared_ptr<Eigen::VectorXd> fullMeanStore,
    std::shared_ptr<std::vector<double>> poseTrailTimeStampsStore)
{
    const Eigen::VectorXd &mean = ekf.getState();
    const Eigen::MatrixXd &cov = ekf.getStateCovarianceRef();

    inertialMean = mean.segment(0, INER_DIM);
    inertialCovDiag = cov.block(0, 0, INER_DIM, INER_DIM).diagonal();
    positionCov = cov.block<3, 3>(POS, POS);
    velocityCov = cov.block<3, 3>(VEL, VEL);

    assert(!inertialMean.hasNaN() && !inertialCovDiag.hasNaN());

    if (poseTrailTimeStampsStore) {
        poseTrailTimeStampsStore->clear();
        const size_t nPosesInState = 1 + (mean.rows() - INER_DIM) / POSE_DIM;
        for (size_t i = 1; i < std::min(stateIndex.poseTrailSize(), nPosesInState); ++i) {
            poseTrailTimeStampsStore->push_back(stateIndex.getTimestamp(i));
        }
    }
    if (fullMeanStore) *fullMeanStore = mean;

    fullMean = fullMeanStore;
    poseTrailTimeStamps = poseTrailTimeStampsStore;
}

void Output::addPoseTrailElementMeanOnly(
    int idx,
    double timestamp,
    const Eigen::Vector3d &pos,
    const Eigen::Vector4d &ori)
{
    assert(idx >= 0);
    if (!fullMean || !poseTrailTimeStamps) return;
    int dim = INER_DIM + (idx + 1) * POSE_DIM;
    if (fullMean->rows() < dim) fullMean->conservativeResize(dim);
    if (int(poseTrailTimeStamps->size()) < idx + 1) {
        assert(int(poseTrailTimeStamps->size()) == idx);
        poseTrailTimeStamps->resize(idx + 1);
    }

    poseTrailTimeStamps->back() = timestamp;
    fullMean->segment<3>(INER_DIM + idx * POSE_DIM) = pos;
    fullMean->segment<4>(INER_DIM + idx * POSE_DIM + 3) = ori;
}

Output::Output() {
    t = 0;
    inertialMean.setZero();
    inertialCovDiag.setZero();
    positionCov.setZero();
    velocityCov.setZero();
    focalLength = -1;
    stationaryVisual = false;
}
}
