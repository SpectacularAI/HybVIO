#ifndef VIEWS_HPP
#define VIEWS_HPP

#include <array>
#include <map>
#include <string>
#include <vector>
#include <opencv2/core/types.hpp>

#include "../api/vio.hpp"
#include "../api/internal.hpp"

// forward declarations
namespace cv { class Mat; }

using PoseHistoryMap = std::map<api::PoseHistory, std::vector<api::Pose>>;
using PoseHistoryPtrMap = std::map<api::PoseHistory, std::vector<api::Pose>*>;

/**
 * Visualizations and other "views" that can work using the APIs.
 * It would be best if most visualizations could work like this without
 * depending on the internals too much.
 */
namespace odometry {
namespace views {

struct MethodOutput {
    cv::Scalar color;
    std::string legend;
    bool shown = true;
    bool exists = false;
};

class PoseOverlayVisualization {
public:
    PoseOverlayVisualization() :
        methodOutputs(),
        alignedBefore(false),
        stickLength(1.0)
    {
        using P = api::PoseHistory;
        methodOutputs[P::OUR] = MethodOutput {
            .color = cv::Scalar(255, 0, 0, 255),
            .legend = "Our"
        };
        methodOutputs[P::GROUND_TRUTH] = MethodOutput {
            .color = cv::Scalar(0, 165, 255, 255),
            .legend = "Ground truth"
        };
        for (auto kind : { P::EXTERNAL, P::ARKIT, P::ARCORE, P::ARENGINE, P::REALSENSE, P::ZED }) {
            methodOutputs[kind] = MethodOutput {
                .color = cv::Scalar(0, 255, 0, 255),
                .legend = "External"
            };
        }
        methodOutputs[P::OUR_PREVIOUS] = MethodOutput {
            .color = cv::Scalar(140, 110, 110, 255),
            .legend = "Our prev"
        };
        methodOutputs[P::GPS] = MethodOutput {
            .color = cv::Scalar(0, 0, 200, 255),
            .legend = "GPS"
        };
        methodOutputs[P::RTK_GPS] = MethodOutput {
            .color = cv::Scalar(255, 255, 255, 255),
            .legend = "RTK-GPS"
        };
    }

    std::map<api::PoseHistory, MethodOutput> methodOutputs;
    bool alignedBefore;
    double stickLength;
};

void visualizePose(
    PoseOverlayVisualization& pov,
    cv::Mat& poseFrame,
    const api::VioApi::VioOutput &output,
    const std::map<int, api::Vector3d> &pointCloudHistory,
    const PoseHistoryMap &poseHistories,
    bool mobileVisualizations,
    bool alignTracks
);

int getIndexWithTime(const std::vector<api::Pose>& poses, double t);

/**
 * Align poses from all histories over the `ref` history. The reference is left unchanged.
 */
void align(PoseHistoryMap &poseHistories, api::PoseHistory ref, bool useWahba = false);
void align(PoseHistoryPtrMap &poseHistories, api::PoseHistory ref, bool useWahba = false);

}
}

#endif
