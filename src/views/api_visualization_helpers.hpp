#ifndef DAZZLING_API_VISUALIZATION_HELPERS
#define DAZZLING_API_VISUALIZATION_HELPERS

#include <map>
#include <vector>

#include "../api/internal.hpp"
#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/standard_ops.hpp>

namespace odometry { struct TaggedFrame; }

namespace api_visualization_helpers {

void trimPoseHistories(PoseHistoryMap &poseHistories);

struct PoseOverlayHandler {
    virtual ~PoseOverlayHandler();

    virtual void setShown(api::PoseHistory poseHistory, bool value) = 0;
    virtual bool getShown(api::PoseHistory poseHistory) const = 0;
    virtual bool getExists(api::PoseHistory poseHistory) const = 0;
};

struct TaggedFrameVisualizer {
    virtual ~TaggedFrameVisualizer();

    virtual std::unique_ptr<accelerated::Image> createDefaultRenderTarget() = 0;
    virtual void visualize(odometry::TaggedFrame &taggedFrame, accelerated::Image &target) = 0;

    static std::unique_ptr<TaggedFrameVisualizer> buildGpu(
        accelerated::Image &colorFrame,
        accelerated::Queue &imageProcessingQueue,
        const api::InternalAPI::DebugParameters &parameters);
};

struct VisualizationHelper : TaggedFrameVisualizer {
    static std::unique_ptr<VisualizationHelper> build(int w, int h,
        accelerated::Queue &imageProcessingQueue,
        const api::InternalAPI::DebugParameters &parameters);

    virtual std::unique_ptr<odometry::TaggedFrame> createTaggedFrame(int tag, std::shared_ptr<accelerated::Image> colorFrame) = 0;
    virtual std::unique_ptr<odometry::TaggedFrame> createTaggedFrameStereo(int tag,
        std::shared_ptr<accelerated::Image> firstRgbaData,
        std::shared_ptr<accelerated::Image> secondRgbaData) = 0;

    virtual void visualizeTaggedFrame(accelerated::Image &target,
            odometry::TaggedFrame &taggedFrame,
            const api::VioApi::VioOutput &output,
            const PoseHistoryMap &poseHistories) = 0;

    virtual void visualizePose(cv::Mat &target,
            const api::VioApi::VioOutput &output,
            const PoseHistoryMap &poseHistories) = 0;

    virtual PoseOverlayHandler &poseOverlayHistory() = 0;
    virtual ~VisualizationHelper();
};

}

#endif
