#include "stereo_upright_2p.hpp"

#include "camera.hpp"
#include "../odometry/triangulation.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"

#include <theia/theia.h>

#include <memory>
#include <vector>

// Theia uses this mechanism to handle the Eigen `std::vector` alignment issues.
#include <theia/alignment/alignment.h>
namespace {
struct Correspondence {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 3d point triangulated from previous frame.
    Eigen::Vector3d modelPoint;
    // Camera ray for current frame.
    Eigen::Vector3d ray;
    // Pre-computed value for error metric.
    Eigen::Vector2d rayNormalized;
};
}
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(Correspondence)

namespace tracker {
namespace {

constexpr size_t MIN_SAMPLE_SIZE = 2;

struct UprightPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Quaterniond rotation;
    Eigen::Vector3d translation;
};

class UprightEstimator : public theia::Estimator<Correspondence, UprightPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    UprightEstimator(const Eigen::Matrix3d worldToCamera) :
        worldToCamera(worldToCamera)
    {}

    double SampleSize() const { return static_cast<double>(MIN_SAMPLE_SIZE); }

    bool EstimateModel(
        const std::vector<Correspondence> &correspondences,
        std::vector<UprightPose> *poses
    ) const {
        const Eigen::Vector3d &modelPoint0 = correspondences[0].modelPoint;
        const Eigen::Vector3d &modelPoint1 = correspondences[1].modelPoint;
        const Eigen::Vector3d &ray0 = correspondences[0].ray;
        const Eigen::Vector3d &ray1 = correspondences[1].ray;
        Eigen::Quaterniond rotations[2];
        Eigen::Vector3d translations[2];
        int solutionCount = theia::TwoPointPosePartialRotation(gravityAxis,
            modelPoint0, modelPoint1, ray0, ray1, rotations, translations);

        poses->clear();
        for (int i = 0; i < solutionCount; ++i) {
            poses->push_back(UprightPose {
                .rotation = rotations[i],
                .translation = translations[i],
            });
        }
        return solutionCount > 0;
    }

    double Error(
        const Correspondence &correspondence,
        const UprightPose &pose
    ) const {
        // Squared reprojection error in the current camera coordinates. This error metric
        // was chosen just because Theia seems to use it for its other pose estimators.
        const Eigen::Vector3d world1 = pose.rotation.matrix() * correspondence.modelPoint + pose.translation;
        const Eigen::Vector2d reprojectedFeature = (worldToCamera * world1).hnormalized();
        return (reprojectedFeature - correspondence.rayNormalized).squaredNorm();
    }

private:
    // The coordinates of the EKF system are defined so that the z-axis is exactly
    // the direction of gravity.
    const Eigen::Vector3d gravityAxis = Eigen::Vector3d(0, 0, 1);
    Eigen::Matrix3d worldToCamera;
};

} // namespace

class StereoUpright2pImplementation: public StereoUpright2p {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoUpright2pImplementation(const odometry::Parameters &parameters) :
        secondToFirstCamera(parameters.imuToCamera * parameters.secondImuToCamera.inverse()),
        parameters()
    {
        const auto &pt = parameters.tracker;
        this->parameters.rng = std::make_shared<theia::RandomNumberGenerator>(pt.ransacRngSeed);
        this->parameters.error_thresh = pt.ransacStereoUpright2pErrorThresh;
        this->parameters.failure_probability = pt.ransacStereoUpright2pFailureProbability;
        this->parameters.max_iterations = pt.ransacStereoUpright2pMaxIterations;
        this->parameters.min_iterations = pt.ransacStereoUpright2pMinIterations;
        this->parameters.min_inlier_ratio = pt.ransacMinInlierFraction;
        this->parameters.use_mle = pt.ransacStereoUpright2pUseMle;
    }

    bool compute(
        const std::vector<std::array<const Camera*, 2>> &cameras,
        const std::vector<std::array<const std::vector<Feature::Point>*, 2>> &corners,
        const std::array<Eigen::Matrix4d, 2> &poses,
        std::vector<Feature::Status> &trackStatus,
        RansacResult &ransacResult
    ) {
        timer(odometry::TIME_STATS, __FUNCTION__);
        const size_t n = trackStatus.size();
        ransacResult.initialize(n);
        assert(corners[0][0]->size() == n);
        assert(corners[0][1]->size() == n);
        assert(corners[1][0]->size() == n);
        assert(corners[1][1]->size() == n);
        Eigen::Matrix3d R0 = poses[0].topLeftCorner<3, 3>();
        Eigen::Matrix3d R1 = poses[1].topLeftCorner<3, 3>();
        correspondences.clear();
        inds.clear();
        for (size_t i = 0; i < n; ++i) {
            // Triangulate 3d points using previous stereo frames.
            // Note that Theia also provides a bunch of triangulation methods in `sfm/triangulation/triangulation.h`.
            if (trackStatus[i] != Feature::Status::TRACKED) continue;
            trackStatus[i] = tracker::Feature::Status::RANSAC_OUTLIER;
            Eigen::Vector2d ip00((*corners[0][0])[i].x, (*corners[0][0])[i].y);
            Eigen::Vector2d ip10((*corners[1][0])[i].x, (*corners[1][0])[i].y);
            Eigen::Vector2d in00, in10;
            if (!cameras[0][0]->normalizePixel(ip00, in00)) continue;
            if (!cameras[1][0]->normalizePixel(ip10, in10)) continue;
            Eigen::Vector3d idp;
            if (!odometry::triangulateStereoFeatureIdp(in00, in10, secondToFirstCamera, idp, nullptr)) {
                continue;
            }
            Eigen::Vector3d modelPoint = Eigen::Vector3d(idp.x(), idp.y(), 1) / idp.z();
            assert(modelPoint[2] > 0.0);

            // Use current left camera frame for rays.
            const size_t cameraInd = 0;
            const Feature::Point &p = (*corners[cameraInd][1])[i];
            Eigen::Vector2d pixel(p.x, p.y);
            Eigen::Vector3d ray;
            if (!cameras[cameraInd][1]->pixelToRay(pixel, ray)) continue;

            // Rotate both pieces of data to their own "world coordinates" which according to
            // the model differ only in 3d translation and rotation around the gravity axis.
            correspondences.push_back({
                .modelPoint = R0 * modelPoint,
                .ray = R1 * ray,
                .rayNormalized = ray.hnormalized(),
            });
            inds.push_back(i);
        }
        if (correspondences.size() < MIN_SAMPLE_SIZE) {
            return false;
        }

        UprightEstimator estimator(R1.inverse());
        std::unique_ptr<theia::SampleConsensusEstimator<UprightEstimator> >
            ransac = CreateAndInitializeRansacVariant(theia::RansacType::RANSAC, parameters, estimator);

        UprightPose uprightPose;
        theia::RansacSummary ransacSummary;
        bool success = ransac->Estimate(correspondences, &uprightPose, &ransacSummary);
        if (!success) {
            return false;
        }

        ransacResult.inlierCount = ransacSummary.inliers.size();
        for (int inlier : ransacSummary.inliers) {
            trackStatus[inds[inlier]] = tracker::Feature::Status::TRACKED;
        }

        ransacResult.type = tracker::RansacResult::Type::UPRIGHT_2P;
        return true;
    }

private:
    const Eigen::Matrix4d secondToFirstCamera;
    theia::RansacParameters parameters;

    // Workspace.
    std::vector<Correspondence> correspondences;
    std::vector<size_t> inds;
};

StereoUpright2p::~StereoUpright2p() = default;

std::unique_ptr<StereoUpright2p> StereoUpright2p::build(const odometry::Parameters &parameters) {
    return std::unique_ptr<StereoUpright2p>(
        new StereoUpright2pImplementation(parameters));
}

}  // namespace tracker
