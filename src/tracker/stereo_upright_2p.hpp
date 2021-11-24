// Implements the two point pose estimation method described in the paper
// “Efficient Computation of Absolute Pose for Gravity-Aware Augmented Reality”
// <https://sites.cs.ucsb.edu/~holl/pubs/Sweeney-2015-ISMAR.pdf>
//
// See also <http://theia-sfm.org/pose.html#two-point-absolute-pose-with-a-partially-known-rotation>.

#ifndef TRACKER_STEREO_UPRIGHT_2P_HPP_
#define TRACKER_STEREO_UPRIGHT_2P_HPP_

#include <vector>

#include "ransac_result.hpp"
#include "track.hpp"

#include <Eigen/Core>

namespace odometry { struct Parameters; }
namespace tracker { class Camera; }

namespace tracker {

class StereoUpright2p {
public:
    static std::unique_ptr<StereoUpright2p> build(const odometry::Parameters &parameters);
    virtual ~StereoUpright2p();

    virtual bool compute(
        const std::vector<std::array<const Camera*, 2>> &cameras,
        const std::vector<std::array<const std::vector<Feature::Point>*, 2>> &corners,
        const std::array<Eigen::Matrix4d, 2> &poses,
        std::vector<Feature::Status> &trackStatus,
        RansacResult &ransacResult
    ) = 0;
};

} // namespace tracker

#endif
