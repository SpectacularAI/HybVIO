#ifndef TRACKER_IMAGE_PYRAMID_H_
#define TRACKER_IMAGE_PYRAMID_H_

#include <accelerated-arrays/image.hpp>
#include <memory>

#include "track.hpp"

namespace accelerated {
    struct Processor;
    namespace operations { struct StandardFactory; }
}

namespace cv { class Mat; }
namespace odometry { struct ParametersTracker; }
namespace tracker {

struct ImagePyramid {
    typedef accelerated::FixedPoint<std::uint8_t> GrayType;
    typedef std::int16_t GradientType;
    static constexpr std::size_t GRADIENT_CHANNELS = 2;
    // what number to multiply the 16-bit integer channel to get
    // the image gradient dI, if the image channels range from 0 to 255
    static constexpr float GRADIENT_SCALE_0_255 = 1.0 / 32;
    // same as above but for channel range 0 to 1
    static constexpr float GRADIENT_SCALE_01 = GRADIENT_SCALE_0_255 / 255;

    virtual accelerated::Image &getGrayLevel(std::size_t i) = 0;
    virtual accelerated::Image &getGradientLevel(std::size_t i) = 0;

    virtual const std::vector<cv::Mat> &getOpenCv() = 0;

    virtual ~ImagePyramid();

    struct Factory {
        virtual std::shared_ptr<ImagePyramid> compute(std::shared_ptr<accelerated::Image> image) = 0;

        virtual ~Factory();

        static std::unique_ptr<Factory> buildOpenCv(const odometry::ParametersTracker &parameters);
    };
};
}

#endif
