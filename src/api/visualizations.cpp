#include "visualizations.hpp"
#include "internal.hpp"
#include <accelerated-arrays/opencv_adapter.hpp>
#include "../views/api_visualization_helpers.hpp"

namespace api {

std::unique_ptr<accelerated::Image> VisualizationVideoOutput::createDefaultRenderTarget() {
    auto visuData = api->getVisualizationHelper();
    assert(visuData);
    return visuData->createDefaultRenderTarget();
}

void VisualizationVideoOutput::update(std::shared_ptr<const VioApi::VioOutput> output) {
    currentOutput = output;
}

void VisualizationVideoOutput::render(cv::Mat &output) {
    std::unique_ptr<accelerated::Image> target = accelerated::opencv::ref(output, true);
    render(*target);
}

void VisualizationVideoOutput::render(accelerated::Image &target) {
    if (!currentOutput) return;
    auto visuData = api->getVisualizationHelper();
    assert(visuData);
    const auto &outImpl = reinterpret_cast<const InternalAPI::Output&>(*currentOutput);
    assert(outImpl.taggedFrame);
    visuData->visualizeTaggedFrame(target, *outImpl.taggedFrame, *currentOutput, outImpl.poseHistories);
};

void VisualizationKfCorrelation::render(cv::Mat &output) {
    api->visualizeKfCorrelation(output);
}

void VisualizationPose::update(std::shared_ptr<const VioApi::VioOutput> output) {
    currentOutput = output;
}

void VisualizationPose::render(cv::Mat &target) {
    const auto &outImpl = reinterpret_cast<const InternalAPI::Output&>(*currentOutput);
    auto visuData = api->getVisualizationHelper();
    assert(visuData);
    visuData->visualizePose(target, *currentOutput, outImpl.poseHistories);
}

bool VisualizationPose::ready() const {
    // This may not be ready at the time the pose visualization is first wanted
    return !!api->getVisualizationHelper();
}

void VisualizationCovarianceMagnitudes::render(cv::Mat &target) {
    api->visualizeCovarianceMagnitudes(target);
}

} // namespace api
