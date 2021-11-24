#ifndef ODOMETRY_POSE_TRAIL_H_
#define ODOMETRY_POSE_TRAIL_H_

#include <unordered_map>
#include <map>
#include <random>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "parameters.hpp"
#include "util.hpp"
#include "triangulation.hpp"

namespace odometry {

struct VisualizationTrack {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > points;
    bool active;
};
using VisualizationTrackCollection = std::map<int, VisualizationTrack, std::less<int>,
         Eigen::aligned_allocator<std::pair<const int, VisualizationTrack> > >;

/** 2D (stereo) feature in a certain frame */
struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // TODO: change to 3D camera ray
        Eigen::Vector2d imagePoint;
        Eigen::Vector2d normalizedImagePoint;
        // Velocity in `normalizedImagePoint` units per seconds, estimated from successive frames.
        Eigen::Vector2d normalizedVelocity;

        Frame() :
            imagePoint(-1, -1),
            normalizedImagePoint(-1, -1),
            normalizedVelocity(0, 0)
        {}
    };
    std::array<Frame, 2> frames;

    // inverse-depth parametrized
    Eigen::Vector3d triangulatedStereoPointIdp;
    Eigen::Matrix3d triangulatedStereoCov;

    bool usedForVisualUpdate;

    Feature() :
        frames(),
        triangulatedStereoPointIdp(0, 0, 0),
        triangulatedStereoCov(Eigen::Matrix3d::Zero()),
        usedForVisualUpdate(false)
    {}
};

struct KeyFrame {
    int frameNumber;
    double timestamp;

    // maps track ID to feature
    std::unordered_map<int, Feature,
        std::hash<int>, std::equal_to<int>,
        Eigen::aligned_allocator<std::pair<const int, Feature> > > features;

    bool hasFeature(int trackId) const {
        return features.count(trackId) > 0;
    }

    void insertFeatureUnlessExists(int trackId, const Feature &feature
    ) {
        if (features.count(trackId) == 0) {
            features[trackId] = feature;
        }
    }
};

class EKFStateIndex {
private:
    const odometry::ParametersOdometry &parameters;
    std::vector<KeyFrame> keyframes;
    int frameCounter = 0;
    /**
     * 3D map points in the EKF state (hybrid EKF-SLAM approach).
     * Each element is a feature track ID (which is also used as the
     * map point ID). Id -1 means empty / unused
     */
    std::vector<int> mapPoints;
    std::vector<int> tmpIndex;

    // remove the "least useful" keyframe and return the index from
    // which it was removed
    int removeKeyframe();

    size_t maxSize() { return parameters.cameraTrailLength + 1; }

public:
    EKFStateIndex(const odometry::Parameters &params) :
        parameters(params.odometry),
        mapPoints(parameters.hybridMapSize, -1)
    {
        assert(parameters.cameraTrailHanoiLength + parameters.cameraTrailStridedLength + 1 < int(maxSize()));
        assert(parameters.randomTrackSamplingRatio > 0.0 && parameters.randomTrackSamplingRatio <= 1.0);
        pushHeadKeyframe(0, 0);
    }

    bool canPopKeyframe() const { return keyframes.size() >= 2; }

    /** @return the index of the least useful removed keyframe */
    int pushHeadKeyframe(int frameNumber, double timestamp);
    void popHeadKeyframe();

    float trackScore(int trackId, TrackSampling selection) const;

    void createTrackIndex(
        int trackId,
        std::vector<int> &index,
        TrackSampling selection,
        std::mt19937 &rng
    );

    /**
     * @param index The output: indices 0,1,2,...,poseCount
     */
    void createFullIndex(std::vector<int> &index) const;

    void markTrackUsed(
        int trackId,
        const std::vector<int> &index,
        TrackSampling selection
    );

    /**
     * Remove all keyframes that do not share any tracks with the current frame.
     * Also remove all map points not visible in the current frame
     */
    void prune();
    /**
     * Offer a new point to the state. Accepted if there are free slots.
     * If accepted, returns the state idx. Otherwise returns -1
     */
    int offerMapPoint(int trackId);

    /** Create index: track Id -> map point ID */
    void createMapPointIndex(std::map<int, int> &index) const {
        index.clear();
        for (std::size_t i = 0; i < mapPoints.size(); ++i) {
            index[mapPoints.at(i)] = i;
        }
    }

    KeyFrame &headKeyFrame() { return keyframes.at(0); }

    /**
     * @param trackId track ID
     * @param imagePoint (output) "current" pixel coordinates, if available.
     *  This may be the head or the first pose trail (non-head) entry
     * @return true iff the the output was set
     */
    bool getCurrentTrackPixelCoordinates(int trackId, Eigen::Vector2f &imagePoint) const;

    void buildTrackVectors(
        int trackId,
        const std::vector<int> &index,
        vecVector2d &imageFeatures,
        vecVector2d &featureVelocities,
        Eigen::VectorXd &y,
        bool stereo
    ) const;

    std::size_t poseTrailSize() const { return keyframes.size(); }

    /**
     * Writes to the reference arguments the feature coordinates and the keyframe indicies
     * corresponding to the two keyframes with widest separation that contain the given track.
     *
     * @param trackId Track whose features are searched.
     * @param kefyrame0 Index of first found keyframe.
     * @param kefyrame1 Index of second found keyframe.
     * @param imagePoint0 First image point.
     * @param imagePoint0 Second image point.
     * @return True if the pose trail contains two separate features associated with the given track id.
     */
    bool widestBaseline(
        int trackId,
        size_t &keyframe0,
        size_t &keyframe1,
        Eigen::Vector2d &imagePoint0,
        Eigen::Vector2d &imagePoint1) const;

    void extract3DFeatures(
        int trackId,
        const std::vector<int> &index,
        CameraPoseTrail &camPoseTrail) const;

    /**
     * Get the tracker frame number of a certain pose trail element by
     * its index in the pose trail.
     */
    int getFrameNumber(int index) const {
        return keyframes.at(index).frameNumber;
    }

    double getTimestamp(int index) const {
        return keyframes.at(index).timestamp;
    }

    /**
     * Get coordinates of tracks present in the most recent keyframe.
     */
    void getVisualizationTracks(VisualizationTrackCollection &tracks) const;

    void updateVelocities(int trackId);
};

} // namespace odometry

#endif
