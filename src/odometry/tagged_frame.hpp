#ifndef DAZZLING_TAGGED_FRAME_H_
#define DAZZLING_TAGGED_FRAME_H_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "../tracker/track.hpp"
#include "output.hpp"

#include "util.hpp"
#include "ekf_state_index.hpp"

namespace accelerated { struct Image; }
namespace tracker { struct Image; }
namespace odometry {
struct TrackVisualization {
    odometry::PrepareVuStatus prepareVuStatus;
    odometry::TriangulatorStatus triangulateStatus;
    bool visualUpdateSuccess;
    bool blacklisted;
    Eigen::VectorXd trackProjection;
    Eigen::VectorXd trackTracker;
    Eigen::VectorXd secondTrackProjection;
    Eigen::VectorXd secondTrackTracker;
};

// Additional data that the end application can supply to odometry with every frame (pair of stereo frames).
// `colorFrame` is a RGBA image (or horizontally concatenated images in stereo case; in very specific format) that the algorithm can draw
//   on to produce visualizations.
// `tag` can be used as a running index to keep track of which TaggedFrame input
//   corresponds to which TaggedFrame output (as odometry doesn't modify the tag).
struct TaggedFrame {
    // Non-optional.
    std::shared_ptr<accelerated::Image> colorFrame;
    int tag;

    // Access to tracker internals for visualization. These are also
    // stored elsewhere for actual processing
    std::shared_ptr<const tracker::Image> firstGrayFrame;
    std::shared_ptr<const tracker::Image> secondGrayFrame;

    // Odometry visual update track visualizations.
    std::vector<TrackVisualization> trackVisualizations;
    // Tracker track visualizations.
    VisualizationTrackCollection trackerTracks;

    std::vector<tracker::Feature::Point> corners;
    // present only in stereo mode
    std::vector<tracker::Feature::Point> secondCorners;
    // For optical flow visualization.
    std::vector<tracker::Feature::Point> flowCorners0;
    std::vector<tracker::Feature::Point> flowCorners1;
    std::vector<tracker::Feature::Point> flowCorners2;
    std::vector<tracker::Feature::Status> flowStatus;
    // For stereo epipolar visualization.
    std::vector<tracker::Feature::Point> epipolarCorners0;
    std::vector<tracker::Feature::Point> epipolarCorners1;
    std::vector<std::vector<tracker::Feature::Point>> epipolarCurves;

    // just references on some part of colorframe
    cv::Rect firstImageRect;
    cv::Rect secondImageRect;

    // used in SLAM: the reprojections of all currently visible SLAM
    // map points (the slamPointCloud)
    std::vector<tracker::Feature::Point> slamPointReprojections;
    // for each corner: the index of this track in slamPointReprojections or
    // -1 if this track does not correspond to any SLAM map point
    std::vector<int> cornerSlamPointIndex;
};

}

#endif
