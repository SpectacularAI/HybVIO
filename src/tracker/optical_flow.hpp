#ifndef TRACKER_OPTICAL_FLOW_H_
#define TRACKER_OPTICAL_FLOW_H_

#include <functional>
#include <memory>
#include <vector>
#include <accelerated-arrays/image.hpp>

#include "track.hpp"
#include "image_pyramid.hpp"

namespace accelerated {
    struct Processor;
    namespace operations { struct StandardFactory; }
}
namespace cv { class Mat; }
namespace odometry { struct ParametersTracker; }

namespace tracker {
struct OpticalFlow {
    static std::unique_ptr<OpticalFlow> buildOpenCv(const odometry::ParametersTracker &p);

    virtual ~OpticalFlow();

    /**
     * Compute optical flow using the Lucas-Kanade method.
     *
     * @param trackStatus output value with Feature::TRACKED and FAILED values.
     *   previous values will be erased.
     */
    virtual void compute(
        ImagePyramid &prevImagePyramid,
        ImagePyramid &imagePyramid,
        const std::vector<Feature::Point> &prevCorners,
        std::vector<Feature::Point> &corners,
        std::vector<Feature::Status> &trackStatus,
        bool useInitialCorners,
        int overrideMaxIterations = -1) = 0;
};
}

#endif
