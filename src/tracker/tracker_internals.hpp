#ifndef TRACKER_INTERNALS_H_
#define TRACKER_INTERNALS_H_

#include "tracker.hpp"
#include "util.hpp"

#include <functional>
#include <unordered_map>
#include <memory>
#include <random>
#include <set>

namespace tracker {
class Camera;
class FeatureDetector;
class OpticalFlow;
class RansacPipeline;
struct RansacResult;

class TrackerImplementation : public Tracker {
public:
    TrackerImplementation(const odometry::Parameters &parameters);
    ~TrackerImplementation();

    void add(const TrackerArgsIn &args, Output &output) final;

    void deleteTrack(int id) final;

    const odometry::Parameters parameters;

    std::vector<Feature::Point> prevCorners, prevSecondCorners;
    std::vector<Feature> tracks;
    int frameNum;
    double prevFrameTime = -1.0;

    // for visualization purposes
    const RansacResult &lastRansacResult() const;

    std::function<void(
        int trackId,
        Feature::Status trackStatus,
        float x,
        float y)> trackLogCallback;
private:
    void initialize(Image& firstImage, Image* secondImage,
        Output &output);
    void track(
        Image& image,
        Image* secondImage,
        std::vector<Feature::Point>& corners,
        std::vector<Feature::Point>& secondCorners,
        const TrackerArgsIn &args,
        Output &output);
    void updateTracks(
        const std::vector<Feature::Point>& firstCorners,
        const std::vector<Feature::Point>& secondCorners,
        std::vector<Feature::Status> &trackStatus,
        std::vector<Feature> &outputTracks,
        bool keyframe);
    void detectNewFeatures(Image& image, Image* secondImage, Output &output);
    void resetAllTracks(const std::vector<Feature::Point>& firstCorners, const std::vector<Feature::Point>& secondCorners);
    void detectFeatures(Image& image, Image* secondImage,
        std::vector<Feature::Point>& corners, std::vector<Feature::Point>& secondCorners,
        Output &output);
    bool computeVisualStationarity(
        const std::vector<Feature::Point> &corners,
        const std::vector<Feature::Status> &trackStatus,
        double ransacStationarityScore,
        double t) const;
    bool isPointInCrop(const Image &image, Feature::Point point);
    void markOutOfDetectionCropCornersAsFailed(
            const Image& image,
            const std::vector<Feature::Point>& corners,
            std::vector<Feature::Status> &trackStatus);
    void markCornersFailedByEpipolarConstraint(
        const std::vector<Feature::Point> &corners0,
        const std::vector<Feature::Point> &corners1,
        const Image &image,
        const Image &secondImage,
        std::vector<Feature::Status> &trackStatus,
        Output &output);
    void computeDenseStereoDepth(
        Image& image,
        Image& secondImage,
        Output &output) const;
    void setMask(const std::vector<Feature::Point> &corners, const std::vector<Feature::Status> &status);
    void changeMaskSize(double change);
    int maskRadius(const Image &forImage) const;
    void logTracks();

    int nextTrackId;
    std::unordered_map<int, Feature::Point> lastKeyframeCornerByTrackId;
    std::shared_ptr<Image> prevImage, prevSecondImage;
    std::unique_ptr<RansacPipeline> ransac;

    std::vector<Feature::Point> maskCorners;
    double maskScale = 0.0;

    struct Distance {
        double dist2;
        size_t i;
        size_t j;
    };

    struct Work {
        // keep std::vectors and Images that are potentially modified
        // on each frame here to avoid unnecessary memory reallocation
        // all of these could be local variables
        std::vector<Feature::Point> corners, secondCorners;
        std::vector<Feature::Status> detectionStatus;
        std::vector<Feature::Status> trackStatus;
        std::vector<Feature::Status> trackStatusStereo;
        std::vector<size_t> brokenTracks;
        std::vector<Distance> distances;
        std::set<size_t> deleteTrackInds;
        std::shared_ptr<Image> curImage, curSecondImage;
        std::vector<tracker::Feature::Point> epipolarCurve;
        std::vector<std::array<const Camera*, 2>> allCameras;
        std::vector<std::array<const std::vector<Feature::Point>*, 2>> allCorners;
    };

    Work workspace;
};
}

#endif
