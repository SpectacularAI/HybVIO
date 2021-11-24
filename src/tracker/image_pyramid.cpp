#include <accelerated-arrays/opencv_adapter.hpp>
#include <opencv2/opencv.hpp>

#include "image_pyramid.hpp"
#include "../odometry/parameters.hpp"
#include "../util/allocator.hpp"

namespace tracker {
namespace {
struct CpuImagePyramid : ImagePyramid {
    std::vector<cv::Mat> cvPyramid;

    const std::vector<cv::Mat> &getOpenCv() final {
        return cvPyramid;
    }

    accelerated::Image &getGrayLevel(std::size_t i) final {
        (void)i;
        assert(false && "TODO");
    }

    accelerated::Image &getGradientLevel(std::size_t i) final {
        (void)i;
        assert(false && "TODO");
    }
};

class CpuImagePyramidFactory : public ImagePyramid::Factory {
private:
    const odometry::ParametersTracker &parameters;
    util::Allocator<CpuImagePyramid> pyramidAllocator;

public:
    CpuImagePyramidFactory(const odometry::ParametersTracker& p)
    :
        parameters(p),
        pyramidAllocator([]() { return std::make_unique<CpuImagePyramid>(); })
    {}

    std::shared_ptr<ImagePyramid> compute(std::shared_ptr<accelerated::Image> img) final {
        auto pyramid = pyramidAllocator.next();
        cv::buildOpticalFlowPyramid(
            accelerated::opencv::ref(*img),
            pyramid->cvPyramid,
            cv::Size(parameters.pyrLKWindowSize, parameters.pyrLKWindowSize),
            parameters.pyrLKMaxLevel);
        return pyramid;
    }
};
}

std::unique_ptr<ImagePyramid::Factory> ImagePyramid::Factory::buildOpenCv(const odometry::ParametersTracker &p) {
    return std::unique_ptr<ImagePyramid::Factory>(new CpuImagePyramidFactory(p));
}

ImagePyramid::~ImagePyramid() = default;
ImagePyramid::Factory::~Factory() = default;

}
