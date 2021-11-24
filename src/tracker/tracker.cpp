#include <algorithm>
#include <cmath>
#include <cassert>
#include <Eigen/Core>

#include "image.hpp"
#include "camera.hpp"
#include "tracker_internals.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"
#include "ransac_pipeline.hpp"
#include "../odometry/util.hpp"

// private helper functions
namespace {
inline double computeDist2(const tracker::Feature::Point &p1, const tracker::Feature::Point &p2) {
    const double dx = p1.x - p2.x, dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

double computeMaxPixelCoordinateMovement(
        const std::vector<tracker::Feature::Point> &corners,
        const std::vector<tracker::Feature> &tracks,
        const std::vector<tracker::Feature::Status> &trackStatus,
        const std::unordered_map<int, tracker::Feature::Point> &lastKeyframeCornerByTrackId) {
    double maxDist = 0;
    int n = 0;
    assert(corners.size() == trackStatus.size());
    for (std::size_t i = 0; i < corners.size(); ++i) {
        if (trackStatus[i] == tracker::Feature::Status::TRACKED) {
            const auto it = lastKeyframeCornerByTrackId.find(tracks[i].id);
            if (it == lastKeyframeCornerByTrackId.end()) continue;
            const auto &prev = it->second;
            const double d = std::sqrt(computeDist2(corners[i], prev));
            maxDist = std::max(maxDist, d);
            n++;
        }
    }
    if (n == 0) return -1.0; // Signal failure.
    return maxDist;
}

void computeOpticalFlow(
    const odometry::ParametersTracker &parameters,
    tracker::Image &prevImage,
    tracker::Image &image,
    const std::vector<tracker::Feature::Point> &prevCorners,
    std::vector<tracker::Feature::Point> &corners,
    std::vector<tracker::Feature::Point> &flowCorners,
    const std::vector<tracker::Feature> &tracks,
    std::vector<tracker::Feature::Status> &trackStatus,
    const tracker::OpticalFlowPredictor &opticalFlowPredictor,
    tracker::OpticalFlowPredictor::Type predictionType,
    bool visualizedKind
) {
    timer(odometry::TIME_STATS, __FUNCTION__);

    bool useInitialCorners = false;
    if (opticalFlowPredictor) {
        if (parameters.predictOpticalFlow) {
            useInitialCorners = true;
            opticalFlowPredictor.func(prevCorners, tracks, corners, predictionType);
        }
        // Save predictions for visualization.
        if (parameters.saveOpticalFlow == odometry::OpticalFlowVisualization::PREDICT
                && visualizedKind) {
            opticalFlowPredictor.func(prevCorners, tracks, flowCorners, predictionType);
        }
    }

    // Compute optical flow for visualization, without using prediction.
    if (parameters.saveOpticalFlow == odometry::OpticalFlowVisualization::COMPARE
            && visualizedKind) {
        std::vector<tracker::Feature::Status> statusCopy = trackStatus;
        image.opticalFlow(prevImage, prevCorners, flowCorners, statusCopy, false);
    }

    image.opticalFlow(prevImage, prevCorners, corners, trackStatus, useInitialCorners);
}

// Empty curve denotes failure.
void computeEpipolarCurve(
    const tracker::Feature::Point &p,
    const tracker::Camera &camera0,
    const tracker::Camera &camera1,
    const Eigen::Matrix4d cam0ToCam1,
    std::vector<tracker::Feature::Point> &curve
) {
    constexpr size_t CURVE_POINTS = 8;
    curve.clear();
    curve.reserve(CURVE_POINTS);
    Eigen::Vector2d pixel(p.x, p.y);
    Eigen::Vector3d ray;
    bool success = camera0.pixelToRay(pixel, ray);
    if (success) {
        float s = 0.5;
        for (size_t j = 0; j < CURVE_POINTS; ++j) {
            Eigen::Vector3d r0 = s * ray;
            Eigen::Vector3d r1 = odometry::util::transformVec3ByMat4(cam0ToCam1, r0);
            success = camera1.rayToPixel(r1, pixel);
            if (!success) break;
            curve.push_back({ static_cast<float>(pixel(0)), static_cast<float>(pixel(1)) });
            s *= 2;
        }
    }
    if (!success) curve.clear();
}

// For visualization.
std::vector<std::vector<tracker::Feature::Point>> computeEpipolarCurves(
    const std::vector<tracker::Feature::Point> &corners0,
    const tracker::Camera &camera0,
    const tracker::Camera &camera1,
    const odometry::Parameters &parameters
) {
    std::vector<std::vector<tracker::Feature::Point>> epipolarCurves;
    epipolarCurves.reserve(corners0.size());
    std::vector<tracker::Feature::Point> curve;
    const Eigen::Matrix4d T = parameters.secondImuToCamera * parameters.imuToCamera.inverse();
    for (tracker::Feature::Point p : corners0) {
        curve.clear();
        computeEpipolarCurve(p, camera0, camera1, T, curve);
        epipolarCurves.push_back(curve);
    }
    return epipolarCurves;
}

bool withinDistanceFromCurve(
    tracker::Feature::Point p,
    const std::vector<tracker::Feature::Point> &curve,
    float dist
) {
    float dist2 = std::pow(dist, 2);
    assert(!curve.empty());
    Eigen::Vector2f pp(p.x, p.y);
    // Reverse iterate because with `computeEpipolarCurves` the point
    // is usually in the "far end".
    for (auto it = curve.rbegin(); it != curve.rend(); ++it) {
        Eigen::Vector2f d(it->x - p.x, it->y - p.y);
        if (d.squaredNorm() < dist2) return true;
    }
    for (size_t i = 0; i + 1 < curve.size(); ++i) {
        Eigen::Vector2f c0(curve[i].x, curve[i].y);
        Eigen::Vector2f c1(curve[i + 1].x, curve[i + 1].y);
        float s2 = (c1 - c0).squaredNorm();
        float t = (pp - c0).dot(c1 - c0) / s2;
        // Check projection to line spanned by the two points is on the segment,
        // and that distance to the projection is below the threshold.
        if (t > 0 && t < 1 && (pp - (c0 + t * (c1 - c0))).squaredNorm() < dist2) {
            return true;
        }
    }
    return false;
}
} // anonymous namespace

namespace tracker {

// base class methods
Tracker::~Tracker() = default; // a virtual dtor is good to have
std::unique_ptr<Tracker> Tracker::build(const odometry::Parameters &parameters) {
    return std::make_unique<TrackerImplementation>(parameters);
}

TrackerImplementation::TrackerImplementation(const odometry::Parameters &parameters) :
    parameters(parameters),
    frameNum(0),
    // private:
    nextTrackId(1)
{
    tracks.reserve(parameters.tracker.maxTracks);

    changeMaskSize(0.0);
}

TrackerImplementation::~TrackerImplementation() = default;

void TrackerImplementation::add(
    const TrackerArgsIn &args,
    tracker::Tracker::Output &output
) {
    // TODO: storing by reference here is questionable, since the Image
    // operations are not necessarily thread-safe
    if (!prevImage) {
        prevImage = args.firstImage;
        prevSecondImage = args.secondImage;
    }
    workspace.curImage = args.firstImage;
    auto &firstImage = *workspace.curImage;
    Image *secondImage = nullptr;
    if (args.secondImage) {
        workspace.curSecondImage = args.secondImage;
        // stereo mode
        secondImage = workspace.curSecondImage.get();
        assert(firstImage.width == secondImage->width && firstImage.height == secondImage->height);
    }

    workspace.corners.clear();
    nextTrackId = frameNum * parameters.tracker.maxTracks + 1;

    if (frameNum == 0) {
        initialize(firstImage, secondImage, output);
        prevFrameTime = args.t;
        return;
    }

    frameNum++;

    if (prevCorners.size() >= 5) {
        // The tracking algorithms need at least five feature points to work with.
        // RANSAC, which is the only component using the camera(s) in tracker,
        // currently only use the first one in stereo mode
        track(
            firstImage,
            secondImage,
            workspace.corners,
            workspace.secondCorners,
            args,
            output
        );
    }
    else {
        setMask({}, {}); // clear mask
        detectFeatures(firstImage, secondImage,
            workspace.corners, workspace.secondCorners, output);
        resetAllTracks(workspace.corners, workspace.secondCorners);
        prevCorners = workspace.corners;
        prevSecondCorners = workspace.secondCorners;
    }

    prevFrameTime = args.t;

    // note that the original image data has already been copied at this point
    std::swap(workspace.curImage, prevImage);
    if (args.secondImage) {
        std::swap(workspace.curSecondImage, prevSecondImage);
    }
    logTracks();
}

void TrackerImplementation::detectFeatures(
    Image& image, Image* secondImage,
    std::vector<Feature::Point>& corners, std::vector<Feature::Point>& secondCorners,
    Output &output
) {
    const auto &camera0 = *image.getCamera();
    const odometry::ParametersTracker &pt = parameters.tracker;
    {
        timer(odometry::TIME_STATS, "findKeypoints");
        image.findKeypoints(maskCorners, maskRadius(image), corners);
    }

    bool stereo = secondImage != nullptr;
    if (stereo) {
        std::vector<Feature::Point> flowCorners;
        computeOpticalFlow(
            pt,
            image,
            *secondImage,
            corners,
            secondCorners,
            flowCorners,
            // No associated tracks.
            {},
            workspace.detectionStatus,
            // Although triangulation is not possible in this case, optical flow estimation
            // could still be used with assumed distance for the 3d feature points.
            // Based on few tests, there seemed to be no clear advantage.
            {},
            OpticalFlowPredictor::Type::STEREO,
            false
        );

        const auto &camera1 = *secondImage->getCamera();
        if (pt.maxStereoEpipolarDistance > 0.0) {
            markCornersFailedByEpipolarConstraint(
                corners,
                secondCorners,
                image,
                *secondImage,
                workspace.detectionStatus,
                output);
        }

        if (pt.saveStereoEpipolar == odometry::StereoEpipolarVisualization::DETECTED) {
            output.epipolarCorners0 = corners;
            output.epipolarCorners1 = secondCorners;
            output.epipolarCurves = computeEpipolarCurves(corners, camera0, camera1, parameters);
        }
    } else {
        workspace.detectionStatus.assign(corners.size(), Feature::Status::TRACKED);
    }

    markOutOfDetectionCropCornersAsFailed(image, corners, workspace.detectionStatus);
    if (stereo) {
        markOutOfDetectionCropCornersAsFailed(*secondImage, secondCorners, workspace.detectionStatus);
    }

    int p = 0;
    for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
        if (workspace.detectionStatus[i] == Feature::Status::TRACKED)
        {
            corners[p] = corners[i];
            if (stereo) secondCorners[p] = secondCorners[i];
            ++p;
        }
    }
    corners.resize(p);
    if (stereo) secondCorners.resize(p);

    if (corners.size() == 0) return;
}


bool TrackerImplementation::isPointInCrop(const Image &image, Feature::Point point) {
    const odometry::ParametersTracker &pt = parameters.tracker;
    double x_delta = image.width * (1 - pt.partOfImageToDetectFeatures) / 2;
    double y_delta = image.height * (1 - pt.partOfImageToDetectFeatures) / 2;
    return point.x >= x_delta && point.x < image.width - x_delta && point.y >= y_delta && point.y < image.height - y_delta;
}

void TrackerImplementation::markOutOfDetectionCropCornersAsFailed(
        const Image& image,
        const std::vector<Feature::Point>& corners,
        std::vector<Feature::Status> &trackStatus)
{
    if (parameters.tracker.fisheyeCamera) {
        Eigen::Vector3d ray;
        for (size_t i = 0; i < corners.size(); i++) {
            auto ip = corners.at(i);
            Eigen::Vector2d pixel(ip.x, ip.y);
            // Test if ray angle is larger than `parameters.tracker.validCameraFov` allows.
            if (!image.getCamera()->pixelToRay(pixel, ray)) {
                trackStatus[i] = Feature::Status::OUT_OF_RANGE;
            }
        }
    }

    if (parameters.tracker.partOfImageToDetectFeatures < 1.0) {
        for (size_t i = 0; i < corners.size(); i++) {
            if (!isPointInCrop(image, corners[i])) {
                trackStatus[i] = Feature::Status::OUT_OF_RANGE;
            }
        }
    }
}

void TrackerImplementation::markCornersFailedByEpipolarConstraint(
    const std::vector<Feature::Point> &corners0,
    const std::vector<Feature::Point> &corners1,
    const Image &image,
    const Image &secondImage,
    std::vector<Feature::Status> &trackStatus,
    Output &output
) {
    const odometry::ParametersTracker &pt = parameters.tracker;
    if (pt.maxStereoEpipolarDistance <= 0) return;
    float imageScale = static_cast<float>(std::min(image.width, image.height));
    const float dist = pt.maxStereoEpipolarDistance * imageScale / 720.0;
    assert(corners0.size() == corners1.size());
    assert(corners0.size() == trackStatus.size());
    const Eigen::Matrix4d T = parameters.secondImuToCamera * parameters.imuToCamera.inverse();
    for (size_t i = 0; i < corners1.size(); ++i) {
        if (trackStatus[i] != tracker::Feature::Status::TRACKED) continue;
        std::vector<tracker::Feature::Point> &curve = workspace.epipolarCurve;
        computeEpipolarCurve(corners0[i], *image.getCamera(), *secondImage.getCamera(), T, curve);
        if (!curve.empty() && !withinDistanceFromCurve(corners1[i], curve, dist)) {
            if (pt.saveStereoEpipolar == odometry::StereoEpipolarVisualization::FAILED) {
                output.epipolarCorners0.push_back(corners0[i]);
                output.epipolarCorners1.push_back(corners1[i]);
                output.epipolarCurves.push_back(curve);
            }
            trackStatus[i] = tracker::Feature::Status::FAILED_EPIPOLAR_CHECK;
        }
    }
}

void TrackerImplementation::track(
    Image& image,
    Image* secondImage,
    std::vector<Feature::Point>& corners,
    std::vector<Feature::Point>& secondCorners,
    const TrackerArgsIn &args,
    Output &output
) {
    const auto &camera0 = *image.getCamera();

    const odometry::ParametersTracker &pt = parameters.tracker;
    auto &trackStatus = workspace.trackStatus;
    using FlowCornerVec = std::vector<Feature::Point>;
    bool useStereo = secondImage != nullptr;

    FlowCornerVec flowCorners;
    assert(prevImage);
    computeOpticalFlow(
        pt,
        *prevImage,
        image,
        prevCorners,
        corners,
        flowCorners,
        tracks,
        trackStatus,
        args.opticalFlowPredictor,
        OpticalFlowPredictor::Type::LEFT,
        // Visualize if mono.
        !useStereo
    );
    if (useStereo) {
        auto &trackStatusStereo = workspace.trackStatusStereo;
        const auto &camera1 = *secondImage->getCamera();
        assert(secondImage && prevSecondImage);
        assert(prevSecondCorners.size() == corners.size());
        tracker::OpticalFlowPredictor::Type predictionType = OpticalFlowPredictor::Type::STEREO;
        std::vector<tracker::Feature::Point> *corners0 = nullptr;
        tracker::Image *image0 = nullptr;
        if (pt.independentStereoOpticalFlow) {
            predictionType = OpticalFlowPredictor::Type::RIGHT;
            image0 = &*prevSecondImage;
            corners0 = &prevSecondCorners;
        }
        else {
            image0 = &image;
            corners0 = &corners;
        }
        computeOpticalFlow(
            pt,
            *image0,
            *secondImage,
            *corners0,
            secondCorners,
            flowCorners,
            tracks,
            trackStatusStereo,
            args.opticalFlowPredictor,
            predictionType,
            true
        );

        assert(trackStatus.size() == trackStatusStereo.size());
        for (size_t i = 0; i < trackStatus.size(); ++i) {
            if (trackStatusStereo[i] == Feature::Status::FAILED_FLOW) {
                trackStatus[i] = Feature::Status::FAILED_FLOW;
            }
        }

        // Does not seem to work well with independentStereoOpticalFlow, so disable if it's enabled.
        if (pt.maxStereoEpipolarDistance > 0.0 && !pt.independentStereoOpticalFlow) {
            markCornersFailedByEpipolarConstraint(
                *corners0,
                secondCorners,
                *image0,
                *secondImage,
                trackStatus,
                output);
        }

        if (pt.saveStereoEpipolar == odometry::StereoEpipolarVisualization::TRACKED) {
            output.epipolarCorners0 = corners;
            output.epipolarCorners1 = secondCorners;
            output.epipolarCurves = computeEpipolarCurves(*corners0, camera0, camera1, parameters);
        }
    }

    markOutOfDetectionCropCornersAsFailed(image, corners, trackStatus);
    if (useStereo) {
        markOutOfDetectionCropCornersAsFailed(*secondImage, secondCorners, trackStatus);
    }

    // Mark previously blacklisted tracks.
    assert(corners.size() == tracks.size());
    assert(corners.size() == trackStatus.size());
    assert(corners.size() == prevCorners.size());
    for (size_t i = 0; i < corners.size(); i++) {
        if (tracks[i].status == Feature::Status::BLACKLISTED) {
            trackStatus[i] = Feature::Status::BLACKLISTED;
        }
    }

    if (useStereo) {
        const auto &camera1 = *secondImage->getCamera();
        workspace.allCameras = {{&camera0, &camera0}, {&camera1, &camera1}};
        workspace.allCorners = {{&prevCorners, &corners}, {&prevSecondCorners, &secondCorners}};
    }
    else {
        workspace.allCameras = {{&camera0, &camera0}};
        workspace.allCorners = {{&prevCorners, &corners}};
    }
    const double ransacStationarityScore =
        ransac->compute(workspace.allCameras, workspace.allCorners, args.poses, trackStatus);

    setMask(corners, trackStatus);
    assert(prevCorners.size() == tracks.size());

    // Gather visualization data. Before `updateTracks()` which deletes bad tracks.
    using V = odometry::OpticalFlowVisualization;
    if (pt.saveOpticalFlow == V::PREDICT || pt.saveOpticalFlow == V::COMPARE) {
        if (useStereo) {
            assert(flowCorners.size() == secondCorners.size());
            if (pt.independentStereoOpticalFlow) {
                assert(prevSecondCorners.size() == secondCorners.size());
                output.flowCorners0 = reinterpret_cast<const FlowCornerVec&>(prevSecondCorners);
            }
            else {
                assert(corners.size() == secondCorners.size());
                output.flowCorners0 = reinterpret_cast<const FlowCornerVec&>(corners);
            }
            output.flowCorners1 = reinterpret_cast<const FlowCornerVec&>(secondCorners);
        }
        else {
            assert(prevCorners.size() == corners.size());
            assert(flowCorners.size() == corners.size());
            output.flowCorners0 = reinterpret_cast<const FlowCornerVec&>(prevCorners);
            output.flowCorners1 = reinterpret_cast<const FlowCornerVec&>(corners);
        }
        output.flowCorners2 = flowCorners;
        output.flowStatus = trackStatus;
    }
    else if (pt.saveOpticalFlow == V::FAILURES) {
        output.flowCorners0 = reinterpret_cast<const FlowCornerVec&>(prevCorners);
        output.flowCorners1 = reinterpret_cast<const FlowCornerVec&>(corners);
    }

    // visual stationarity / keyframe selection may interfere with other
    // initialization mechanisms in odometry: accept all first frames to get
    // a full pose history ASAP
    output.keyframe = frameNum < pt.maxTrackLength ||
        !computeVisualStationarity(corners, trackStatus, ransacStationarityScore, args.t);

    updateTracks(corners, secondCorners, trackStatus, output.tracks, output.keyframe);
    detectNewFeatures(image, secondImage, output);
    if (useStereo && pt.computeDenseStereoDepth) computeDenseStereoDepth(image, *secondImage, output);

    // After `updateTracks()` to also record `CULLED` track statuses.
    if (pt.saveOpticalFlow == V::FAILURES) {
        output.flowStatus = trackStatus;
    }

    // Tune feature detector mask size to improve number of features and their spread.
    size_t maxTracks = static_cast<size_t>(pt.maxTracks);
    if (tracks.size() < (3 * maxTracks) / 4) {
        changeMaskSize(-1.0);
    }
    else if (tracks.size() == maxTracks) {
        changeMaskSize(0.5);
    }

    // Construct prevCorners for use on next frame.
    prevCorners.clear();
    for (size_t i = 0; i < tracks.size(); i++) {
        prevCorners.push_back(tracks[i].points[0]);
    }
    if (useStereo) {
        prevSecondCorners.clear();
        for (size_t i = 0; i < tracks.size(); i++) {
            prevSecondCorners.push_back(tracks[i].points[1]);
        }
    }
}

void TrackerImplementation::changeMaskSize(double change) {
    maskScale += change;
    const double minScale = -5.0;
    const double maxScale = 5.0;
    if (maskScale < minScale) maskScale = minScale;
    if (maskScale > maxScale) maskScale = maxScale;
}

int TrackerImplementation::maskRadius(const Image &forImage) const {
    const double step = 1.3;
    double scale = std::pow(step, maskScale);
    const int minDim = std::min(forImage.width, forImage.height);
    int r = std::round(scale * minDim * parameters.tracker.relativeMaskRadius);
    if (r < 2) r = 2;
    return r;
}

bool TrackerImplementation::computeVisualStationarity(
    const std::vector<Feature::Point> &corners,
    const std::vector<Feature::Status> &trackStatus,
    double ransacStationarityScore,
    double t
) const {
    const odometry::ParametersTracker &pt = parameters.tracker;
    double deltaTime = 0.0;
    if (prevFrameTime > 0.0) {
        deltaTime = t - prevFrameTime;
    }

    assert(prevImage);
    const double maxMovement = computeMaxPixelCoordinateMovement(corners, tracks, trackStatus, lastKeyframeCornerByTrackId);
    if (maxMovement < 0.0) return false;

    // note: using raw pixel coordinates as the maximum theshold
    const double threshold = pt.visualStationarityMovementThreshold;
    // log_debug("visual stationarity score %g, threshold %g", maxMovement, threshold);
    const double stationarityScore = ransacStationarityScore
        * (maxMovement < threshold ? 1.0 : 0.0);

    const bool isStationary = stationarityScore > pt.visualStationarityScoreThreshold;
    return isStationary;
}

void TrackerImplementation::updateTracks(
        const std::vector<Feature::Point> &firstCorners,
        const std::vector<Feature::Point> &secondCorners,
        std::vector<Feature::Status> &trackStatus,
        std::vector<Feature> &outputTracks,
        bool keyframe)
{
    const odometry::ParametersTracker &pt = parameters.tracker;
    bool stereoMode = secondCorners.size() > 0;
    assert(!stereoMode || firstCorners.size() == secondCorners.size());

    outputTracks.clear();

    // Delete some of the closest tracks if we are at maximum capacity.
    // Since this coincides with increasing detector mask size, it should help
    // spread the features better over the whole frame.
    size_t maxTracks = static_cast<size_t>(pt.maxTracks);
    if (firstCorners.size() == maxTracks) {
        workspace.distances.clear();
        workspace.deleteTrackInds.clear();
        for (size_t i = 0; i < firstCorners.size(); ++i) {
            for (size_t j = i + 1; j < firstCorners.size(); ++j) {
                double dist2 = computeDist2(firstCorners[i], firstCorners[j]);
                workspace.distances.push_back({ dist2, i, j });
            }
        }
        std::stable_sort(workspace.distances.begin(), workspace.distances.end(), [](Distance a, Distance b) { return a.dist2 < b.dist2; });
        for (const auto &d : workspace.distances) {
            // Larger id is probably the newer and shorter track.
            workspace.deleteTrackInds.insert(d.j);
            trackStatus[d.j] = Feature::Status::CULLED;
            if (workspace.deleteTrackInds.size() > maxTracks / 20) {
                break;
            }
        }
    }

    // Collect new segments into tracks.
    workspace.brokenTracks.clear();
    auto &broken_tracks = workspace.brokenTracks;
    for (size_t i = 0; i < firstCorners.size(); i++) {
        tracks[i].status = trackStatus[i];
        if (trackStatus[i] == Feature::Status::TRACKED) {
            tracks[i].points[0] = firstCorners[i];
            // TODO: last keyframe points
            if (stereoMode) {
                tracks[i].points[1] = secondCorners[i];
            }

            if (trackStatus[i] != Feature::Status::BLACKLISTED) {
                outputTracks.push_back(tracks[i]);
                if (keyframe)
                    lastKeyframeCornerByTrackId[tracks[i].id] = tracks[i].points[0];
            }
        }
        else {
            // Mark track for deletion.
            broken_tracks.push_back(i);
            lastKeyframeCornerByTrackId.erase(tracks[i].id);
        }
    }

    // Erase the broken tracks using reverse iteration.
    for (auto i = broken_tracks.rbegin(); i != broken_tracks.rend(); ++i) {
        tracks.erase(tracks.begin() + static_cast<int>(*i));
    }
}

void TrackerImplementation::detectNewFeatures(Image& firstImage, Image* secondImage, Output &output)
{
    // Detect new features to replace broken tracks. Might not replace all.
    // This uses the features in decreasing order of quality.

    size_t maxTracks = static_cast<size_t>(parameters.tracker.maxTracks);
    assert(tracks.size() <= maxTracks);
    size_t missing = maxTracks - tracks.size();

    const bool stereo = secondImage != nullptr;

    // Feature detection is the slowest part of the algorithm so skip it when
    // we already have almost all the track spots filled. In future we might
    // want to investigate alternative feature detection algorithms.
    if (missing >= maxTracks / 10) {
        detectFeatures(firstImage, secondImage,
            workspace.corners, workspace.secondCorners, output);

        std::size_t collected_tracks = 0;
        for (size_t i = 0; i < workspace.corners.size() && collected_tracks < missing; ++i) {
            Feature track { .id = nextTrackId };
            track.points[0] = workspace.corners[i];
            if (stereo) {
                track.points[1] = workspace.secondCorners[i];
            }
            tracks.push_back(track);
            ++collected_tracks, ++nextTrackId;
        }
    }
    // It's normal if tracks.size() < parameters.tracker.maxTracks.
    assert(tracks.size() <= maxTracks);
}

void TrackerImplementation::resetAllTracks(const std::vector<Feature::Point>& firstCorners, const std::vector<Feature::Point>& secondCorners) {
    assert(firstCorners.size() == secondCorners.size() || secondCorners.empty());
    tracks.clear();
    lastKeyframeCornerByTrackId.clear();
    for (size_t i = 0; i < firstCorners.size(); i++) {
        Feature track { .id = nextTrackId };
        track.points[0] = firstCorners[i];
        if (!secondCorners.empty()) { // stereo
            track.points[1] = secondCorners[i];
        }
        tracks.push_back(track);
        nextTrackId++;
    }
    assert(tracks.size() <= static_cast<size_t>(parameters.tracker.maxTracks));
}

// API for deleting tracks, eg based on results from odometry.
// Because the deleted track can at the earliest be replaced at
// the end of processing the next frame, we can just mark the
// track here and do the actual deleting during processing that
// next frame.
void TrackerImplementation::deleteTrack(int id) {
    // Because of buffering of camera frames, the call to delete a track
    // usually comes a frame late so that blacklisted tracks don't disappear
    // immediately and often there is an attempt to delete the same
    // track twice, ie this loop finds no match because the track was
    // already deleted.
    for (auto& track : tracks) {
        if (track.id == id) {
            track.status = Feature::Status::BLACKLISTED;
            return;
        }
    }
}

void TrackerImplementation::logTracks() {
    if (trackLogCallback) {
        for (const auto &track : tracks) {
            const auto &p = track.points[0];
            trackLogCallback(track.id, track.status, p.x, p.y);
        }
    }
}

void TrackerImplementation::initialize(Image& firstImage, Image* secondImage, Output &output)
{
    ransac = RansacPipeline::build(firstImage.width, firstImage.height, parameters);

    detectFeatures(firstImage, secondImage,
        workspace.corners, workspace.secondCorners, output);
    resetAllTracks(workspace.corners, workspace.secondCorners);

    prevCorners = workspace.corners;
    prevSecondCorners = workspace.secondCorners;
    frameNum = 1;
    logTracks();

    output.tracks.clear();
    output.keyframe = true;
}

void TrackerImplementation::setMask(
    const std::vector<Feature::Point> &corners,
    const std::vector<Feature::Status> &status
) {
    assert(corners.size() == status.size());
    maskCorners.clear();
    for (std::size_t i = 0; i < corners.size(); ++i) {
        if (status[i] == Feature::Status::TRACKED) {
            maskCorners.push_back(corners[i]);
        }
    }
}

const RansacResult &TrackerImplementation::lastRansacResult() const {
    assert(ransac);
    return ransac->lastResult();
}

void TrackerImplementation::computeDenseStereoDepth(
    Image& image,
    Image& secondImage,
    Output &output) const
{
    const auto &pt = parameters.tracker;
    assert(pt.useRectification);
    image.computeDisparity(secondImage);
    if (pt.computeStereoPointCloud) image.getStereoPointCloud();
    for (auto &track : output.tracks) {
        track.depth = image.getDepth(Eigen::Vector2f(track.points[0].x, track.points[0].y));
    }
}
} // namespace tracker
