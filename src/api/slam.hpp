/* Odometry-SLAM API for plugging in an external SLAM module */
#ifndef SLAM_API_H
#define SLAM_API_H

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <memory>
#include <utility>
#include <vector>
#include <future>
#include "../tracker/track.hpp"

// forward declarations
namespace odometry { struct Parameters; }
namespace tracker { struct Image; }
namespace cv { class Mat; }

namespace slam {

struct DebugAPI;

struct Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix4d pose;
    // Uncertainty matrix
    Eigen::Matrix<double, 3, 6> uncertainty;
    double t;
    int frameNumber;
};

using tracker::Feature;

class Slam {
public:
    struct Result {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        struct MapPoint {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            /** Unique ID of the map point */
            int id;
            /**
             * If this map point corresponds to an input track, the id of that track,
             * otherwise -1
             */
            int trackId;
            /** Global position of the map point */
            Eigen::Vector3d position;
        };

        using PointCloud = std::vector<MapPoint, Eigen::aligned_allocator<MapPoint>>;

        Eigen::Matrix4d poseMat;
        PointCloud pointCloud;
    };

    static std::unique_ptr<Slam> build(const odometry::Parameters &parameters);
    virtual ~Slam() = default;

    /**
     * @param frame gray camera frame (mono / first camera)
     * @param poseTrail VIO pose trail matching the camera frame
     * @param features list of current tracked features in the camera frame
     * @param camera camera model corresponding to the frame
     * @param colorFrame for visualizations & debug. Can be empty
     * @return The SLAM-corrected position of this frame & the local point cloud
     *   as a future
     */
    virtual std::future<Result> addFrame(
        std::shared_ptr<tracker::Image> frame,
        const std::vector<slam::Pose> &poseTrail,
        const std::vector<Feature> &features,
        const cv::Mat &colorFrame) = 0;

    /**
     * Signal SLAM to perform actions on session end. (e.g. map saving)
     */
    virtual std::future<bool> end() = 0;

    /**
     * Connect to internals for debugging
     */
    virtual void connectDebugAPI(DebugAPI &debugAPI) = 0;
};

}

#endif
