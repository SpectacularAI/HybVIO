#ifndef TRACKER_H_
#define TRACKER_H_

#include "track.hpp"
#include "../odometry/parameters.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace tracker {
struct Image;

struct OpticalFlowPredictor {
    enum class Type {
        // From previous left camera pose to current left camera pose.
        LEFT,
        // From previous right camera pose to current right camera pose.
        RIGHT,
        // From current left camera pose to current right camera pose.
        STEREO,
    };

    std::function< void(
        const std::vector<Feature::Point>&,
        const std::vector<Feature>&,
        std::vector<Feature::Point> &out,
        Type type
    ) > func;

    inline operator bool() const {
        return !!func;
    }
};

struct TrackerArgsIn {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::shared_ptr<Image> firstImage;
    // Can be `nullptr`.
    std::shared_ptr<Image> secondImage;
    // Timestamp.
    double t;
    const OpticalFlowPredictor &opticalFlowPredictor;
    // World-to-camera transformation for previous and current left camera frames.
    const std::array<Eigen::Matrix4d, 2> *poses = nullptr;
};

/**
 * The tracker API. Only use the tracker through this interface
 * in odometry
 */
class Tracker {
public:
    struct Output {
        std::vector<Feature> tracks;
        /**
         * If true, the new feature locations have been appended
         * to the tracks. If not, they have replaced the last feature
         * points on each track
         */
        bool keyframe;
        // For visualizations.
        std::vector<Feature::Point> flowCorners0;
        std::vector<Feature::Point> flowCorners1;
        std::vector<Feature::Point> flowCorners2;
        std::vector<Feature::Status> flowStatus;
        std::vector<Feature::Point> epipolarCorners0;
        std::vector<Feature::Point> epipolarCorners1;
        std::vector<std::vector<Feature::Point>> epipolarCurves;
    };

    /**
     * Feed a frame to the tracker.
     */
    virtual void add(const TrackerArgsIn &args, Output &output) = 0;

    virtual void deleteTrack(int id) = 0;

    virtual ~Tracker();

    // build a concrete tracker implementation
    static std::unique_ptr<Tracker> build(const odometry::Parameters &parameters);
};

} // namespace tracker

#endif // TRACKER_H_
