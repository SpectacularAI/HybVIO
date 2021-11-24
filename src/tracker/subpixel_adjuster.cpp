#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "subpixel_adjuster.hpp"
#include "../odometry/parameters.hpp"

namespace tracker {
namespace {
class SubPixelAdjusterImplementation : public SubPixelAdjuster {
private:
    const odometry::ParametersTracker &parameters;
    std::vector<cv::Point2f> tmpCorners; // workspace

public:
    SubPixelAdjusterImplementation(const odometry::ParametersTracker &p)
    : parameters(p) {}

    void adjust(Image& trackerImage, std::vector<Feature::Point>& features) final {
        auto &corners = reinterpret_cast<std::vector<cv::Point2f>&>(features);
        // the OpenCV method assert that the input is non-empty, which
        // crashes the program without this check
        if (corners.empty()) return;

        const cv::Mat &image = reinterpret_cast<tracker::CpuImage&>(trackerImage).getOpenCvMat();

        // Refine corner locations.
        tmpCorners = corners;
        cv::cornerSubPix(image, corners,
            cv::Size(parameters.subPixWindowSize, parameters.subPixWindowSize),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                parameters.subPixMaxIter,
                parameters.subPixEpsilon));

        // Reject refined locations outside the image boundaries and use the original locations.
        for (size_t i = 0; i < corners.size(); i++) {
            cv::Point2f point = corners[i];
            if (point.x < 0.0f || point.x >= image.cols || point.y < 0.0f || point.y >= image.rows) {
                corners[i] = tmpCorners[i];
            }
        }
    }
};
}

SubPixelAdjuster::~SubPixelAdjuster() = default;
std::unique_ptr<SubPixelAdjuster> SubPixelAdjuster::build(const odometry::ParametersTracker &p) {
    return std::unique_ptr<SubPixelAdjuster>(new SubPixelAdjusterImplementation(p));
}
}
