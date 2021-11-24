#include "../api/slam.hpp"
#include "../util/logging.hpp"
#include "../odometry/parameters.hpp"

using namespace Eigen;

namespace slam {

std::unique_ptr<Slam> Slam::build(const odometry::Parameters &parameters) {
    if (parameters.slam.useSlam) {
        log_error("Option `-useSlam` requires building with option `-DUSE_SLAM=ON`.");
    }
    return nullptr;
}

}
