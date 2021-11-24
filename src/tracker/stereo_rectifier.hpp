#ifndef TRACKER_STEREO_RECTIFIER_H_
#define TRACKER_STEREO_RECTIFIER_H_

#include <array>
#include <memory>
#include <Eigen/Dense>

namespace accelerated { struct Image; }
namespace api { struct CameraParameters; }
namespace odometry { struct Parameters; }

namespace tracker {
class Camera;

class StereoRectifier {
public:
    using Image = accelerated::Image;
    static std::unique_ptr<StereoRectifier> build(
        int imageWidth, int imageHeight,
        const std::array<api::CameraParameters, 2> &intrinsics,
        const odometry::Parameters &parameters);

    virtual ~StereoRectifier();
    virtual std::array<std::shared_ptr<const Camera>, 2> getRectifiedCameras() const = 0;
    virtual Eigen::Matrix4d getDepthQMatrix() const = 0;
};
}

#endif
