#include <opencv2/opencv.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

#include "optical_flow.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"

namespace tracker {
namespace {
void computeImplementationCpu(
    const odometry::ParametersTracker &parameters,
    std::vector<float> &err,
    std::vector<std::uint8_t> &charTrackStatus,
    const std::vector<cv::Mat> &prevPyramid,
    const std::vector<cv::Mat> &imagePyramid,
    const std::vector<Feature::Point> &prevCornersNoCv,
    std::vector<Feature::Point> &cornersNoCv,
    std::vector<Feature::Status> &trackStatus,
    bool useInitialCorners
) {
    const auto &prevCorners = reinterpret_cast<const std::vector<cv::Point2f>&>(prevCornersNoCv);
    auto &corners = reinterpret_cast<std::vector<cv::Point2f>&>(cornersNoCv);
    const auto &img = imagePyramid.at(0);
    const int width = img.cols, height = img.rows;

    err.clear(); // ignored
    trackStatus.clear();
    trackStatus.resize(static_cast<size_t>(prevCorners.size()), Feature::Status::FAILED_FLOW);
    charTrackStatus.clear();
    charTrackStatus.resize(static_cast<size_t>(prevCorners.size()), 0);

    // Predict flow using odometry.
    int flags = 0;
    if (useInitialCorners) {
        flags |= cv::OPTFLOW_USE_INITIAL_FLOW;
    }

    if (prevCorners.empty()) {
        corners.clear();
        return;
    }

    const cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            parameters.pyrLKMaxIter, parameters.pyrLKEpsilon);

    cv::calcOpticalFlowPyrLK(prevPyramid, imagePyramid, prevCorners, corners,
            charTrackStatus, err,
            cv::Size(parameters.pyrLKWindowSize, parameters.pyrLKWindowSize),
            parameters.pyrLKMaxLevel, termcrit, flags, parameters.pyrLKMinEigThreshold);
    assert(corners.size() == prevCorners.size());

    for (size_t i = 0; i < trackStatus.size(); i++) {
        cv::Point2f point = corners[i];
        trackStatus[i] = charTrackStatus[i] == 0 ? Feature::Status::FAILED_FLOW : Feature::Status::TRACKED;
        if (point.x < 0.0f || point.x >= width || point.y < 0.0f || point.y >= height) {
            trackStatus[i] = Feature::Status::FLOW_OUT_OF_RANGE;
        }
    }
}

class OpenCvOpticalFlow : public OpticalFlow {
private:
    // workspace / ignored
    std::vector<float> workErr;
    std::vector<std::uint8_t> charTrackStatus;

    const odometry::ParametersTracker &parameters;
    odometry::ParametersTracker overrideParameters;

public:
    OpenCvOpticalFlow(const odometry::ParametersTracker& p)
    :
        parameters(p)
    {
        overrideParameters = parameters;
    }

    void compute(
        ImagePyramid &prevImagePyramid,
        ImagePyramid &imagePyramid,
        const std::vector<Feature::Point> &prevCornersNoCv,
        std::vector<Feature::Point> &cornersNoCv,
        std::vector<Feature::Status> &trackStatus,
        bool useInitialCorners,
        int overrideMaxIterations
    ) final {
        const odometry::ParametersTracker *currentParams = &parameters;
        if (overrideMaxIterations > 0) {
            overrideParameters.pyrLKMaxIter = overrideMaxIterations;
            currentParams = &overrideParameters;
        }
        computeImplementationCpu(
            *currentParams,
            workErr,
            charTrackStatus,
            prevImagePyramid.getOpenCv(),
            imagePyramid.getOpenCv(),
            prevCornersNoCv,
            cornersNoCv,
            trackStatus,
            useInitialCorners);
    }
};
}

std::unique_ptr<OpticalFlow> OpticalFlow::buildOpenCv(const odometry::ParametersTracker& p) {
    return std::unique_ptr<OpticalFlow>(new OpenCvOpticalFlow(p));
}

OpticalFlow::~OpticalFlow() = default;
}
