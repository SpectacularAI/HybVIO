#ifndef TRACKER_SUBPIXEL_ADJUSTER_H_
#define TRACKER_SUBPIXEL_ADJUSTER_H_

#include "track.hpp"
#include <memory>

namespace odometry { struct ParametersTracker; }
namespace tracker {
struct Image;

struct SubPixelAdjuster {
    static std::unique_ptr<SubPixelAdjuster> build(const odometry::ParametersTracker &parameters);
    virtual ~SubPixelAdjuster();
    virtual void adjust(Image& image, std::vector<Feature::Point>& corners) = 0;
};
}

#endif
