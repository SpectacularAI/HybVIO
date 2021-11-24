#include "ransac_pipeline.hpp"
#include "camera.hpp"
#include "track.hpp"
#include "../odometry/triangulation.hpp" // Ransac3
#include "../odometry/parameters.hpp"
#include "ransac_result.hpp"
#include "rot_ransac.hpp"
#include "five_point.hpp"
#include "stereo_upright_2p.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"

#include <theia/theia.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <memory>

namespace tracker {
void RansacResult::initialize(size_t nTrackedFeatures) {
    type = Type::SKIPPED;
    inlierCount = 0;
    inliers.clear();
    inliers.resize(nTrackedFeatures, tracker::Feature::Status::RANSAC_OUTLIER);
    // NOTE: leaves the cv::Mats in an undefined state
    // TODO: use cv::Matx33f etc. instead
}

void RansacResult::updateTrackStatus(std::vector<Feature::Status> &trackStatus) const {
    if (type == Type::SKIPPED) return;

    size_t j = 0;
    for (size_t i = 0; i < trackStatus.size(); i++) {
        if (trackStatus[i] != tracker::Feature::Status::TRACKED) continue;
        const bool isInlier = inliers.at(j) == tracker::Feature::Status::TRACKED;
        if (!isInlier) trackStatus[i] = tracker::Feature::Status::RANSAC_OUTLIER;
        j++;
    }
    assert(j == inliers.size());
}

namespace {
class RansacPipelineImplementation : public RansacPipeline {
private:
    const odometry::Parameters &parameters;
    std::mt19937 rng;
    rot_ransac::RotRansac rotRansac;
    std::unique_ptr<StereoUpright2p> stereoUpright2p;
    RansacResult ransacResult;
    RansacResult ransac2Result, ransac5Result;
    theia::RansacParameters theiaRansac5Parameters, ransac3Parameters;

    // workspace
    std::vector<Feature::Point> c1, c2;
    std::vector<double> r5data1, r5data2;
    std::vector<theia::FeatureCorrespondence> correspondences;
    std::vector<int> validPixelIndex;
    std::vector<std::uint8_t> tmpInliers;
    std::vector<theia::FeatureCorrespondence2D3D> ransac3Correspondences;
    std::vector<size_t> ransac3Inds;

public:
    RansacPipelineImplementation(int width, int height, const odometry::Parameters& parameters)
    :
        parameters(parameters),
        rng(parameters.tracker.ransacRngSeed)
    {
        const auto &pt = parameters.tracker;
        // <http://www.theia-sfm.org/ransac.html>
        theiaRansac5Parameters.rng = std::make_shared<theia::RandomNumberGenerator>(pt.ransacRngSeed);
        theiaRansac5Parameters.error_thresh = pt.theiaRansac5ErrorThresh;
        theiaRansac5Parameters.failure_probability = pt.theiaRansac5FailureProbability;
        theiaRansac5Parameters.max_iterations = pt.theiaRansac5MaxIterations;
        theiaRansac5Parameters.min_iterations = pt.theiaRansac5MinIterations;
        theiaRansac5Parameters.min_inlier_ratio = pt.ransacMinInlierFraction;
        theiaRansac5Parameters.use_mle = pt.theiaRansac5UseMle;

        ransac3Parameters.rng = std::make_shared<theia::RandomNumberGenerator>(pt.ransacRngSeed);
        ransac3Parameters.error_thresh = pt.ransac3ErrorThresh;
        ransac3Parameters.failure_probability = pt.ransac3FailureProbability;
        ransac3Parameters.max_iterations = pt.ransac3MaxIterations;
        ransac3Parameters.min_iterations = pt.ransac3MinIterations;
        ransac3Parameters.min_inlier_ratio = pt.ransacMinInlierFraction;
        ransac3Parameters.use_mle = pt.ransac3UseMle;

        // Simple scaling factor which works well for 16:9 or more square aspect ratios.
        const double su = std::min(height, width) / 720.0;
        rotRansac.threshold_pow2 = static_cast<float>(std::pow(pt.ransac2Threshold * su, 2));

        if (parameters.tracker.useStereoUpright2p) {
            stereoUpright2p = StereoUpright2p::build(parameters);
        }
    }

    double compute(
        const std::vector<std::array<const Camera*, 2>> &cameras,
        const std::vector<std::array<const std::vector<Feature::Point>*, 2>> &corners,
        const std::array<Eigen::Matrix4d, 2> *poses,
        std::vector<Feature::Status> &trackStatus) final
    {
        for (size_t i = 0; i < corners.size(); ++i) {
            assert(corners[i][0]->size() == trackStatus.size());
            assert(corners[i][1]->size() == trackStatus.size());
        }

        // Pick the left camera features for which track was found.
        c1.clear();
        c2.clear();
        for (size_t i = 0; i < trackStatus.size(); i++) {
            if (trackStatus[i] == Feature::Status::TRACKED) {
                c1.push_back((*corners[0][0])[i]);
                c2.push_back((*corners[0][1])[i]);
            }
        }
        const size_t nTrackedFeatures = c1.size();
        assert(c2.size() == nTrackedFeatures);

        // Run RANSAC2 always as it's used for the stationarity score.
        bool ransac2Done = doRansac2(c1, c2, *cameras[0][0], *cameras[0][1], rotRansac, rng, ransac2Result);

        if (parameters.tracker.useRansac3 && cameras.size() >= 2 && corners.size() >= 2) {
            doRansac3(cameras, corners, trackStatus);
        }
        else if (parameters.tracker.useStereoUpright2p && cameras.size() >= 2
                && corners.size() >= 2 && poses) {
            stereoUpright2p->compute(cameras, corners, *poses, trackStatus, ransacResult);
        }
        else if (parameters.tracker.useHybridRansac) {
            computeHybridRansac(*cameras[0][0], *cameras[0][1], ransac2Done);
            if (ransacResult.type != RansacResult::Type::SKIPPED) {
                ransacResult.updateTrackStatus(trackStatus);
            }
        }
        else {
            return ransac2Result.inlierCount / static_cast<double>(nTrackedFeatures);
        }
        assert(ransacResult.inlierCount <= static_cast<size_t>(parameters.tracker.maxTracks));

        if (ransacResult.type == RansacResult::Type::SKIPPED) {
            // Clear all tracks.
            for (size_t i = 0; i < trackStatus.size(); i++) {
                trackStatus[i] = Feature::Status::RANSAC_OUTLIER;
            }
        }

        if (nTrackedFeatures == 0) {
            return 0.0;
        }
        return ransac2Result.inlierCount / static_cast<double>(nTrackedFeatures);
    }

    // for visualization purposes
    const RansacResult &lastResult() const final {
        return ransacResult;
    }

private:
    void computeHybridRansac(
        const Camera &camera1,
        const Camera &camera2,
        bool ransac2Done
    ) {
        const auto &pt = parameters.tracker;
        const size_t nTrackedFeatures = c1.size();

        // If lots of inliers, skip RANSAC-5.
        const bool useRansac2Inliers = ransac2Result.inlierCount > pt.ransac2InliersToSkipRansac5 * nTrackedFeatures;
        bool ransac5Done = !useRansac2Inliers &&
            doRansac5(pt, c1, c2, camera1, camera2, ransac5Result, theiaRansac5Parameters);

        // Choose RANSAC method to use.

        // Reject tracks with low inlier count.
        double ransac5Fraction = static_cast<double>(ransac5Result.inlierCount) / static_cast<double>(nTrackedFeatures);
        double ransac2Fraction = static_cast<double>(ransac2Result.inlierCount) / static_cast<double>(nTrackedFeatures);
        if (ransac5Fraction < pt.ransacMinInlierFraction) ransac5Done = false;
        if (ransac2Fraction < pt.ransacMinInlierFraction) ransac2Done = false;

        if (ransac2Done && !ransac5Done) {
            ransacResult = ransac2Result;
        }
        else if (ransac5Done && !ransac2Done) {
            ransacResult = ransac5Result;
        }
        else if (ransac2Done && ransac5Done) {
            // This criterion might not be perfect.
            if (useRansac2Inliers || ransac2Result.inlierCount > pt.ransac2InliersOverRansac5Needed * ransac5Result.inlierCount) {
                ransacResult = ransac2Result;
            } else {
                ransacResult = ransac5Result;
            }
        } else {
            ransacResult = RansacResult(); // skipped
        }
    }

    static bool doRansac2(
        const std::vector<Feature::Point> &c1,
        const std::vector<Feature::Point> &c2,
        const Camera &camera1,
        const Camera &camera2,
        rot_ransac::RotRansac &rotRansac,
        std::mt19937 &rng,
        RansacResult &r
    ) {
        timer(odometry::TIME_STATS, __FUNCTION__);
        const size_t nTrackedFeatures = c1.size();
        r.initialize(nTrackedFeatures);

        if (nTrackedFeatures < 2) return false;
        r.R = rotRansac.fit(c1, c2, camera1, camera2, r.inliers, rng);
        r.inlierCount = rotRansac.bestInlierCount;
        r.type = tracker::RansacResult::Type::R2;

        return true;
    }

    bool doRansac3(
        const std::vector<std::array<const Camera*, 2>> &cameras,
        const std::vector<std::array<const std::vector<Feature::Point>*, 2>> &corners,
        std::vector<Feature::Status> &trackStatus
    ) {
        timer(odometry::TIME_STATS, __FUNCTION__);
        size_t nTrackedFeatures = corners[0][0]->size();
        Eigen::Matrix4d T = parameters.imuToCamera * parameters.secondImuToCamera.inverse();
        ransacResult.initialize(nTrackedFeatures);

        // Gather data for the pose estimation problem.
        ransac3Correspondences.clear();
        ransac3Inds.clear();
        for (size_t i = 0; i < nTrackedFeatures; ++i) {
            // Triangulate 3d points using previous stereo frames.
            // Note that Theia also provides a bunch of triangulation methods in `sfm/triangulation/triangulation.h`.
            if (trackStatus[i] != Feature::Status::TRACKED) continue;
            trackStatus[i] = tracker::Feature::Status::RANSAC_OUTLIER;
            Eigen::Vector2d ip00((*corners[0][0])[i].x, (*corners[0][0])[i].y);
            Eigen::Vector2d ip10((*corners[1][0])[i].x, (*corners[1][0])[i].y);
            Eigen::Vector2d in00, in10;
            if (!cameras[0][0]->normalizePixel(ip00, in00)) continue;
            if (!cameras[1][0]->normalizePixel(ip10, in10)) continue;
            theia::FeatureCorrespondence2D3D c;
            Eigen::Vector3d idp;
            if (!odometry::triangulateStereoFeatureIdp(in00, in10, T, idp, nullptr)) continue;
            c.world_point = Eigen::Vector3d(idp.x(), idp.y(), 1) / idp.z();
            assert(c.world_point[2] > 0.0);
            // Use current left camera frame for rays.
            Eigen::Vector2d ip01((*corners[0][1])[i].x, (*corners[0][1])[i].y);
            if (!cameras[0][1]->normalizePixel({ ip01[0], ip01[1] }, c.feature)) continue;
            ransac3Correspondences.push_back(c);
            ransac3Inds.push_back(i);
        }

        if (ransac3Correspondences.size() < 3) {
            return false;
        }

        theia::CalibratedAbsolutePose pose;
        theia::RansacSummary ransacSummary;
        bool success = EstimateCalibratedAbsolutePose(ransac3Parameters, theia::RansacType::RANSAC,
            ransac3Correspondences, &pose, &ransacSummary);
        if (!success) {
            return false;
        }

        ransacResult.inlierCount = ransacSummary.inliers.size();
        for (int inlier : ransacSummary.inliers) {
            trackStatus[ransac3Inds[inlier]] = tracker::Feature::Status::TRACKED;
        }

        ransacResult.type = tracker::RansacResult::Type::R3;
        return true;
    }

    bool doRansac5(
        const odometry::ParametersTracker &parameters,
        const std::vector<Feature::Point> &c1,
        const std::vector<Feature::Point> &c2,
        const Camera &camera1,
        const Camera &camera2,
        RansacResult &r,
        const theia::RansacParameters &theiaRansac5Parameters
    ) {
        timer(odometry::TIME_STATS, __FUNCTION__);
        const size_t nTrackedFeatures = c1.size();
        r.initialize(nTrackedFeatures);

        constexpr size_t MIN_FEATURES = 5;
        if (nTrackedFeatures < MIN_FEATURES) return false;

        if (parameters.useTheiaRansac5) {
            correspondences.clear();
            validPixelIndex.clear();
            for (size_t i = 0; i < nTrackedFeatures; i++) {
                // Normalize the coordinates. OpenCV findEssentialMat() does this internally.
                Eigen::Vector2d p1, p2;

                if (camera1.normalizePixel({ c1[i].x, c1[i].y }, p1) &&
                    camera2.normalizePixel({ c2[i].x, c2[i].y }, p2)) {
                    validPixelIndex.push_back(i);
                    correspondences.push_back(theia::FeatureCorrespondence(p1, p2));
                }
                r.inliers.at(i) = tracker::Feature::Status::RANSAC_OUTLIER;
            }

            theia::RelativePose theiaPose;
            theia::RansacSummary theiaRansacSummary;
            // TODO: use parameters.ransac5Threshold
            bool success = theia::EstimateRelativePose(theiaRansac5Parameters, theia::RansacType::RANSAC, correspondences, &theiaPose, &theiaRansacSummary);
            if (!success) {
                // Not handling failure because it never happens?
                log_debug("Theia RANSAC5 failed.");
            }

            cv::Matx33d essentialMatrix;
            cv::eigen2cv(theiaPose.rotation, r.R);
            cv::eigen2cv(theiaPose.position, r.t);
            cv::eigen2cv(theiaPose.essential_matrix, essentialMatrix);

            for (int inlier : theiaRansacSummary.inliers) {
                r.inliers.at(validPixelIndex.at(inlier)) = tracker::Feature::Status::TRACKED;
            }
        } else {
            // Use a customized version of the cv::findEssentialMat() function that allows control of iteration counts,
            // thereby alleviating worst-case performance.
            // since the RANSAC2 one is too

            // this normalization used to be inside findEssentialMatRansacMaxIter
            // TODO: use image width instead
            const double threshold = 2 * parameters.ransac5Threshold / (camera1.getFocalLength() + camera2.getFocalLength());

            r5data1.clear();
            r5data2.clear();
            r5data1.reserve(nTrackedFeatures * 2);
            r5data2.reserve(nTrackedFeatures * 2);
            validPixelIndex.clear();
            tmpInliers.clear();

            for (std::size_t i = 0; i < nTrackedFeatures; ++i) {
                Eigen::Vector2d p1(c1[i].x, c1[i].y), p2(c2[i].x, c2[i].y);
                Eigen::Vector2d h1, h2;
                if (camera1.normalizePixel(p1, h1) && camera2.normalizePixel(p2, h2)) {
                    r5data1.push_back(h1.x());
                    r5data1.push_back(h1.y());
                    r5data2.push_back(h2.x());
                    r5data2.push_back(h2.y());
                    validPixelIndex.push_back(i);
                }
                r.inliers.at(i) = tracker::Feature::Status::RANSAC_OUTLIER;
            }

            if (validPixelIndex.size() < MIN_FEATURES) return false;
            tmpInliers.resize(validPixelIndex.size(), 1);

            cv::Mat r5points1(validPixelIndex.size(), 1, CV_64FC2, r5data1.data());
            cv::Mat r5points2(validPixelIndex.size(), 1, CV_64FC2, r5data2.data());

            cv::Mat_<double> essentialMatrix = tracker::findEssentialMatRansacMaxIter(
                r5points1, r5points2, parameters.ransac5Prob, threshold,
                tmpInliers, parameters.ransacMaxIters);

            for (std::size_t idx = 0; idx < tmpInliers.size(); ++idx) {
                r.inliers.at(validPixelIndex.at(idx)) = tmpInliers.at(idx) == 0
                    ? Feature::Status::RANSAC_OUTLIER : Feature::Status::TRACKED;
            }

            // The original OpenCV function can be used like this:
            // E = cv::findEssentialMat(c1, c2, focalLength, principalPoint, cv::RANSAC,
            //         ransacProb, parameters.ransac5Threshold, ransac5Inliers);

            // Can performance be significantly improved by handling the non unique
            // cases somehow? Theia's result is unique.
            if (essentialMatrix.rows == 3) {
                // TODO: skipping due to generalized camera model
                // Fix if needed or use Theia
                /*cv::Mat R, t; // TODO: avoid allocation of these, use a "workspace"
                // removed cheiralityInliers which did not do anything here
                cv::recoverPose(essentialMatrix, c1, c2,
                    R, t, focalLength, principalPoint);
                assert(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1);*/
                r.R *= 0;
                r.t *= 0;
            } else {
                log_debug("non-uniqueness in recoverPose");
                r.t *= 0.0; // mark failed operation, a bit hacky
            }
        }

        for (size_t i = 0; i < nTrackedFeatures; i++) {
            if (r.inliers.at(i) == tracker::Feature::Status::TRACKED) {
                r.inlierCount++;
            }
        }

        r.type = tracker::RansacResult::Type::R5;
        return true;
    }
};
}

std::unique_ptr<RansacPipeline> RansacPipeline::build(int w, int h, const odometry::Parameters& p) {
    return std::unique_ptr<RansacPipeline>(new RansacPipelineImplementation(w, h, p));
}
}
