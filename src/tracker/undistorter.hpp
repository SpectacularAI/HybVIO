#ifndef TRACKER_UNDISTORTER_H_
#define TRACKER_UNDISTORTER_H_

#include <Eigen/Dense>
#include <accelerated-arrays/image.hpp>
#include <memory>

namespace accelerated { namespace operations { struct StandardFactory; } }
namespace odometry { struct ParametersTracker; }

namespace tracker {
class Camera;

struct Undistorter {
    struct Result {
        std::shared_ptr<const Camera> camera;
        std::shared_ptr<accelerated::Image> image;
        accelerated::Future future;
    };

    virtual ~Undistorter();

    static std::unique_ptr<Undistorter> buildRectified(
        int width, int height,
        std::shared_ptr<const Camera> rectifiedCamera,
        accelerated::Image::Factory &ifac,
        accelerated::operations::StandardFactory &ofac,
        const odometry::ParametersTracker &parameters);

    static std::unique_ptr<Undistorter> buildMono(
        int width, int height,
        float focalLength,
        accelerated::Image::Factory &ifac,
        accelerated::operations::StandardFactory &ofac,
        const odometry::ParametersTracker &parameters);

    virtual Result undistort(accelerated::Image &image, std::shared_ptr<const Camera> camera) = 0;
};
}

#endif
