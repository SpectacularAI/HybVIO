#include "api_visualization_helpers.hpp"
#include "../api/type_convert.hpp"
#include "../odometry/tagged_frame.hpp"
#include "../tracker/image.hpp"
#include "../util/allocator.hpp"
#include "../util/util.hpp"
#include "../views/views.hpp"
#include "../views/visualization_internals.hpp"
#include <opencv2/opencv.hpp>
#include <accelerated-arrays/future.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

namespace api_visualization_helpers {

namespace {
using OpenCvPointVec = const std::vector<cv::Point2f>;
using api::InternalAPI;
using api::Pose;
using api::Vector3d;

constexpr unsigned MAX_POSITION_HISTORY_LENGTH = 7500;
constexpr unsigned MAX_POINT_CLOUD_HISTORY_LENGTH = 100000;
constexpr unsigned HISTORY_ERASE_BLOCK_SIZE = 200;

const std::array<cv::Scalar, 20> sharpColors = {{
    cv::Scalar(255, 255, 255),
    cv::Scalar(0, 0, 0),
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 255),
    cv::Scalar(128, 0, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(0, 0, 128),
    cv::Scalar(128, 255, 0),
    cv::Scalar(128, 0, 255),
    cv::Scalar(255, 128, 0),
    cv::Scalar(0, 128, 255),
    cv::Scalar(255, 0, 128),
    cv::Scalar(0, 255, 128),
    cv::Scalar(128, 255, 255),
    cv::Scalar(255, 128, 255),
    cv::Scalar(255, 255, 128),
}};

void visualizeCorners(cv::Mat &buf, const odometry::TaggedFrame &taggedFrame, bool bold) {
    const auto &corners = reinterpret_cast<OpenCvPointVec&>(taggedFrame.corners);
    const auto &repro = reinterpret_cast<OpenCvPointVec&>(taggedFrame.slamPointReprojections);

    double scale = static_cast<double>(std::max(buf.cols, buf.rows)) / 1280.0;
    const int rSlam = scale >= 0.5 ? 2 : 1;

    for (const auto &p : repro) {
        const auto color = cv::Scalar(0x0, 0x80, 0xff);
        cv::circle(buf, p, rSlam, color, -1);
    }

    int r = static_cast<int>(std::round(scale * 6.0));
    if (r <= 0) r = 1;

    const int weight = (bold && scale >= 0.75) ? 2 : 1;
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const int slamPointIndex = taggedFrame.cornerSlamPointIndex.at(i);
        auto color = cv::Scalar(0xff, 0x0, 0xff);
        if (slamPointIndex != -1) {
            color = cv::Scalar(0x0, 0xff, 0xff);
            cv::line(buf, corners.at(i), repro.at(slamPointIndex), color);
        }
        cv::circle(buf, corners.at(i), r, color, weight);
    }
}

void visualizeTracks(cv::Mat &buf, const odometry::TaggedFrame &taggedFrame, bool showAll) {
    double scale = static_cast<double>(std::max(buf.cols, buf.rows)) / 1280.0;
    int r = static_cast<int>(std::round(scale * 3.0));
    if (r <= 0) r = 1;
    for (const auto &it : taggedFrame.trackerTracks) {
        if (!showAll && !it.second.active) continue;
        auto color = it.second.active ? cv::Scalar(0xff, 0xff, 0xff) : cv::Scalar(0x99, 0x99, 0x99);
        const auto &p = it.second.points;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            cv::Point2d p0(p[i][0], p[i][1]);
            cv::Point2d p1(p[i + 1][0], p[i + 1][1]);
            cv::line(buf, p0, p1, color);
            cv::circle(buf, p1, r, color);
        }
        if (!p.empty()) {
            cv::Point2d p0(p[0][0], p[0][1]);
            cv::circle(buf, p0, 2*r, color);
        }
    }
}

void visualizeStereoMatching(cv::Mat& colorFrame,
        const std::vector<cv::Point2f>& firstCorners, const std::vector<cv::Point2f>& secondCorners,
        const cv::Rect& firstImageRect, const cv::Rect& secondImageRect) {
    assert (firstCorners.size() == secondCorners.size());
    const cv::Scalar WHITE = cv::Scalar(255, 255, 255, 255);
    for (size_t i = 0; i < firstCorners.size(); ++i) {
        cv::line(
                colorFrame,
                cv::Point2f(firstCorners[i].x + firstImageRect.x, firstCorners[i].y + firstImageRect.y),
                cv::Point2f(secondCorners[i].x + secondImageRect.x, secondCorners[i].y + secondImageRect.y),
                WHITE
        );
    }
}

void visualizeStereoEpipolar(cv::Mat &colorFrame, const odometry::TaggedFrame &taggedFrame) {
    const auto &c0 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.epipolarCorners0);
    const auto &c1 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.epipolarCorners1);
    cv::Point2f d(taggedFrame.secondImageRect.x, 0.0);
    assert(c0.size() == c1.size());
    size_t n = std::min(sharpColors.size(), c0.size());
    for (size_t i = 0; i < n; ++i) {
        if (c0[i].x < 0 || c0[i].x >= d.x) continue;
        if (c0[i].y < 0 || c0[i].y >= taggedFrame.firstImageRect.height) continue;
        if (c1[i].x < 0 || c1[i].x >= colorFrame.cols) continue;
        if (c1[i].y < 0 || c1[i].y >= taggedFrame.secondImageRect.height) continue;
        cv::Scalar color = sharpColors[i];
        cv::circle(colorFrame, c0.at(i), 6, color, 2);
        cv::circle(colorFrame, c1.at(i) + d, 6, color, 2);
        const std::vector<tracker::Feature::Point> &curve = taggedFrame.epipolarCurves[i];
        for (size_t j = 0; j + 1 < curve.size(); ++j) {
            cv::line(colorFrame,
                cv::Point2f(curve[j].x, curve[j].y) + d,
                cv::Point2f(curve[j + 1].x, curve[j + 1].y) + d,
                color);
            cv::rectangle(colorFrame,
                cv::Point2f(curve[j].x - 1, curve[j].y - 1) + d,
                cv::Point2f(curve[j].x + 1, curve[j].y + 1) + d,
                color);
        }
    }
}

void visualizeOpticalFlow(cv::Mat &buf, const odometry::TaggedFrame &taggedFrame) {
    const auto &c0 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.flowCorners0);
    const auto &c1 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.flowCorners1);
    const auto &c2 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.flowCorners2);
    const auto &status = taggedFrame.flowStatus;
    assert(c1.size() == c0.size());
    assert(status.size() == c0.size());
    // Red:   Flow used by odometry.
    // Cyan:  Rejected flow.
    // White: Depending on chosen visualization:
    //        -flow:  Predicted flow (affects actual flow if parameter predictOpticalFlow is true).
    //        -flow2: flow computed without prediction (equal to actual flow if predictOpticalFlow == false).
    auto color1 = cv::Scalar(0x00, 0x00, 0xff);
    auto color2 = cv::Scalar(0xff, 0xff, 0xff);
    auto color3 = cv::Scalar(0xff, 0xff, 0x00);
    double scale = static_cast<double>(std::max(buf.cols, buf.rows)) / 1280.0;
    int r = static_cast<int>(std::round(scale * 6.0));
    if (r <= 0) r = 1;

    assert(c2.size() <= c0.size());
    for (std::size_t i = 0; i < c2.size(); ++i) {
        cv::line(buf, c0.at(i), c2.at(i), color2);
        cv::circle(buf, c2.at(i), r, color2, 1);
    }
    assert(c1.size() <= c0.size());
    for (std::size_t i = 0; i < c1.size(); ++i) {
        auto color = status[i] == tracker::Feature::Status::TRACKED ? color1 : color3;
        cv::line(buf, c0.at(i), c1.at(i), color);
        cv::circle(buf, c1.at(i), r, color, 1);
    }
}

void visualizeOpticalFlowFailures(cv::Mat &buf, const odometry::TaggedFrame &taggedFrame) {
    const auto &c0 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.flowCorners0);
    const auto &c1 = reinterpret_cast<OpenCvPointVec&>(taggedFrame.flowCorners1);
    const auto &status = taggedFrame.flowStatus;
    assert(c1.size() == c0.size());
    assert(status.size() == c0.size());
    double scale = static_cast<double>(std::max(buf.cols, buf.rows)) / 1280.0;
    int r = static_cast<int>(std::round(scale * 6.0));
    if (r <= 0) r = 1;

    cv::Scalar color;
    using Status = tracker::Feature::Status;
    for (std::size_t i = 0; i < c1.size(); ++i) {
        switch (status[i]) {
            case Status::TRACKED: // White
                color = cv::Scalar(0xff, 0xff, 0xff); break;
            case Status::FAILED_FLOW: // Cyan
                color = cv::Scalar(0xff, 0xff, 0x00); break;
            case Status::RANSAC_OUTLIER: // Green
                color = cv::Scalar(0x00, 0xff, 0x00); break;
            case Status::FLOW_OUT_OF_RANGE: // Red
                color = cv::Scalar(0x00, 0x00, 0xff); break;
            case Status::OUT_OF_RANGE: // Dark red
                color = cv::Scalar(0x00, 0x00, 0xaa); break;
            case Status::FAILED_EPIPOLAR_CHECK: // Magenta
                color = cv::Scalar(0xff, 0x00, 0xff); break;
            case Status::CULLED: // Pink
                color = cv::Scalar(0xcb, 0xc0, 0xff); break;
            case Status::BLACKLISTED: // Black
                color = cv::Scalar(0x00, 0x00, 0x00); break;
            case Status::NEW:
                // This case shouldn't happen, `NEW` status only exists inside the tracker.
            default:
                color = cv::Scalar(0x88, 0x88, 0x88); break;
        }
        cv::line(buf, c0.at(i), c1.at(i), color);
        cv::circle(buf, c1.at(i), r, color, 1);
    }
}

struct PoseOverlayHandlerImplementation : PoseOverlayHandler {
    odometry::views::PoseOverlayVisualization poseOverlayVisualization;

    void setShown(api::PoseHistory poseHistory, bool value) final {
        auto &outs = poseOverlayVisualization.methodOutputs;
        // Toggle all external methods together (can use the same key command).
        using P = api::PoseHistory;
        if (poseHistory == P::EXTERNAL) {
            for (auto p : { P::EXTERNAL, P::ARKIT, P::ARCORE, P::ARENGINE }) {
                outs.at(p).shown = value;
            }
        }
        else {
            outs.at(poseHistory).shown = value;
        }
    }

    bool getShown(api::PoseHistory poseHistory) const final {
        const auto &outs = poseOverlayVisualization.methodOutputs;
        return outs.count(poseHistory) && outs.at(poseHistory).shown;
    }

    bool getExists(api::PoseHistory poseHistory) const final {
        const auto &outs = poseOverlayVisualization.methodOutputs;
        return outs.count(poseHistory) && outs.at(poseHistory).exists;
    }
};


struct VisualizationHelperImplementation : VisualizationHelper, private PoseOverlayHandlerImplementation {
    const InternalAPI::DebugParameters &parameters;
    const cv::Size size;
    accelerated::Queue &queue;

    cv::Mat outputFrame;
    std::unique_ptr<accelerated::Image::Factory> stereoImageFactory;
    std::unique_ptr< util::Allocator<accelerated::Image> > stereoFrameBuffer;

    std::unique_ptr<odometry::TaggedFrame> lastTaggedFrame;

    std::map<int, api::Vector3d> pointCloudHistory;
    std::map<int, int> pointCloudSeenCount;

    VisualizationHelperImplementation(int w, int h,
            accelerated::Queue &queue,
            const api::InternalAPI::DebugParameters &parameters) :
        parameters(parameters),
        size(w, h),
        queue(queue)
    {}

    std::unique_ptr<accelerated::Image> createDefaultRenderTarget() final {
        assert(!outputFrame.empty());
        auto out = accelerated::opencv::ref(outputFrame, true); // preferFixedPoint=true
        // Ideally OpenCV RGB -> BGR color flip should be handled here
        return accelerated::cpu::Image::createFactory()->createLike(*out);
    }

    void visualizeTaggedFrame(accelerated::Image &targetImage,
        odometry::TaggedFrame &taggedFrame,
        const api::VioApi::VioOutput &output,
        const PoseHistoryMap &poseHistories) final
    {
        if (targetImage.storageType != accelerated::ImageTypeSpec::StorageType::CPU) {
            log_debug("gpuVisualizer not supported, skipping visualization");
            return;
        }

        visualize(taggedFrame, targetImage);

        if (parameters.visualizePoseOverlay) {
            cv::Mat target = accelerated::opencv::ref(targetImage);
            assert(!target.empty());
            cv::Mat firstImageReference = target;
            if (parameters.api.parameters.tracker.useStereo) {
                firstImageReference = cv::Mat(target, lastTaggedFrame->firstImageRect);
            }
            visualizePose(firstImageReference, output, poseHistories);
        }
    }

    std::unique_ptr<odometry::TaggedFrame> createTaggedFrame(int tag, std::shared_ptr<accelerated::Image> inputColorFrame) final
    {
        assert(inputColorFrame->width == size.width && inputColorFrame->height == size.height);

        accelerated::opencv::copy(*inputColorFrame, outputFrame);

        auto taggedFrame = std::make_unique<odometry::TaggedFrame>();
        taggedFrame->tag = tag;
        taggedFrame->colorFrame = inputColorFrame;
        return taggedFrame;
    }

    std::unique_ptr<odometry::TaggedFrame> createTaggedFrameStereo(int tag,
        std::shared_ptr<accelerated::Image> firstImage,
        std::shared_ptr<accelerated::Image> secondImage) final
    {
        auto taggedFrame = std::make_unique<odometry::TaggedFrame>();
        taggedFrame->tag = tag;

        // handle all stereo visualization on the CPU
        cv::Mat firstRgbaData, secondRgbaData;
        accelerated::opencv::copy(*firstImage, firstRgbaData);
        accelerated::opencv::copy(*secondImage, secondRgbaData);
        queue.processAll();

        cv::Mat colorFrame;
        if (!outputFrame.empty()) colorFrame = outputFrame;

        int rotation = ::util::modulo(parameters.api.parameters.odometry.rot, 4);
        if (rotation == 0) {
            cv::hconcat(firstRgbaData, secondRgbaData, colorFrame);
            taggedFrame->firstImageRect = cv::Rect(0, 0, size.width, size.height);
            taggedFrame->secondImageRect = cv::Rect(size.width, 0, size.width, size.height);
        } else if (rotation == 1) {
            cv::vconcat(secondRgbaData, firstRgbaData, colorFrame);
            taggedFrame->firstImageRect = cv::Rect(0, size.height, size.width, size.height);
            taggedFrame->secondImageRect = cv::Rect(0, 0, size.width, size.height);
        } else if (rotation == 2) {
            cv::hconcat(secondRgbaData, firstRgbaData, colorFrame);
            taggedFrame->firstImageRect = cv::Rect(size.width, 0, size.width, size.height);
            taggedFrame->secondImageRect = cv::Rect(0, 0, size.width, size.height);
        } else if (rotation == 3) {
            cv::vconcat(firstRgbaData, secondRgbaData, colorFrame);
            taggedFrame->firstImageRect = cv::Rect(0, 0, size.width, size.height);
            taggedFrame->secondImageRect = cv::Rect(0, size.height, size.width, size.height);
        }

        if (outputFrame.empty()) {
            outputFrame = colorFrame;
            stereoImageFactory = accelerated::cpu::Image::createFactory();
            stereoFrameBuffer = std::make_unique< util::Allocator<accelerated::Image> >([this]() {
                auto outFrameRef = accelerated::opencv::ref(outputFrame, true); // preferFixedPoint=true
                return stereoImageFactory->createLike(*outFrameRef);
            });
        }
        taggedFrame->colorFrame = stereoFrameBuffer->next();
        accelerated::opencv::copy(colorFrame, *taggedFrame->colorFrame);
        queue.processAll();
        return taggedFrame;
    }

    void visualizePose(cv::Mat &target,
            const api::VioApi::VioOutput &output,
            const PoseHistoryMap &poseHistories) final
    {
        if (parameters.visualizePointCloud) {
            for (const auto &p : output.pointCloud) {
                int count = 1;
                Eigen::Vector3d vec(p.position.x, p.position.y, p.position.z);
                if (pointCloudHistory.count(p.id) > 0) {
                    count = pointCloudSeenCount[p.id] + 1;
                    auto v = pointCloudHistory[p.id];
                    vec = (vec + Eigen::Vector3d(v.x, v.y, v.z) * (count - 1.0)) * (1.0 / count);
                }
                pointCloudHistory[p.id] = api::eigenToVector(vec);
                pointCloudSeenCount[p.id] = count;
            }
            while (pointCloudHistory.size() > MAX_POINT_CLOUD_HISTORY_LENGTH) {
                auto first = pointCloudHistory.begin();
                pointCloudSeenCount.erase(first->first);
                pointCloudHistory.erase(first);
            }
        }
        constexpr bool alignTracks = true; // TODO: legacy from Fusion
        odometry::views::visualizePose(
            poseOverlayVisualization,
            target,
            output,
            pointCloudHistory,
            poseHistories,
            parameters.mobileVisualizations,
            alignTracks
        );
    }

    PoseOverlayHandler &poseOverlayHistory() final {
        return *this;
    }

    void visualize(odometry::TaggedFrame &taggedFrame, accelerated::Image &targetImage) final {
        auto buf = accelerated::opencv::ref(targetImage);
        copyTaggedFrameToCvMat(taggedFrame, buf);
        const auto &pt = parameters.api.parameters.tracker;

        cv::Mat firstImageReference = buf, secondImageReference;
        if (pt.useStereo) {
            firstImageReference = cv::Mat(buf, taggedFrame.firstImageRect);
            secondImageReference = cv::Mat(buf, taggedFrame.secondImageRect);
        }

        typedef InternalAPI::VisualizationMode Mode;
        switch (parameters.visualization) {
            case Mode::PLAIN_VIDEO:
                break;
            case Mode::DEBUG_VISUALIZATION:
                visualizeCorners(firstImageReference, taggedFrame, false);
                for (auto trackVisualization : taggedFrame.trackVisualizations) {
                    visualizeTrack(firstImageReference, trackVisualization, 0);
                    if (pt.useStereo) {
                        visualizeTrack(secondImageReference, trackVisualization, 1);
                    }
                }
                break;
            case Mode::PROCESSED_VIDEO:
                break;
            case Mode::TRACKS:
            case Mode::TRACKS_ALL:
                visualizeTracks(firstImageReference, taggedFrame, parameters.visualization == Mode::TRACKS_ALL);
                break;
            case Mode::OPTICAL_FLOW:
                visualizeOpticalFlow(firstImageReference, taggedFrame);
                break;
            case Mode::OPTICAL_FLOW_FAILURES:
                visualizeOpticalFlowFailures(firstImageReference, taggedFrame);
                break;
            case Mode::TRACKER_ONLY:
                // TODO: re-enable
                break;
            case Mode::CORNER_MEASURE:
                taggedFrame.firstGrayFrame->debugVisualize(
                    firstImageReference,
                    tracker::Image::VisualizationMode::CORNER_MEASURE);
                queue.processAll();
                // NOTE: the corners here are not in sync with the image
                // visualizeCorners(firstImageReference, taggedFrame, false);
                break;
            case Mode::STEREO_MATCHING:
                assert(pt.useStereo);
                visualizeStereoMatching(
                        buf,
                        reinterpret_cast<OpenCvPointVec&>(taggedFrame.corners),
                        reinterpret_cast<OpenCvPointVec&>(taggedFrame.secondCorners),
                        taggedFrame.firstImageRect,
                        taggedFrame.secondImageRect);
                break;
            case Mode::STEREO_EPIPOLAR:
                assert(pt.useStereo);
                visualizeStereoEpipolar(buf, taggedFrame);
                break;
            case Mode::STEREO_DEPTH:
            case Mode::STEREO_DISPARITY:
                taggedFrame.firstGrayFrame->debugVisualize(
                    firstImageReference,
                    parameters.visualization == Mode::STEREO_DEPTH
                        ? tracker::Image::VisualizationMode::DEPTH
                        : tracker::Image::VisualizationMode::DISPARITY);
                visualizeCorners(firstImageReference, taggedFrame, false);
                for (auto trackVisualization : taggedFrame.trackVisualizations) {
                    visualizeTrack(firstImageReference, trackVisualization, 0);
                }
                queue.processAll();
                break;
            case Mode::NONE:
                break;
            default:
                assert(false);
        }
    }

private:
    void copyTaggedFrameToCvMat(odometry::TaggedFrame &taggedFrame, cv::Mat &buf)
    {
        auto &accImg = *taggedFrame.colorFrame;

        assert(!buf.empty());
        assert(buf.channels() == accImg.channels);

        accelerated::opencv::copy(accImg, buf);
        queue.processAll();
    }
};
} // anonymous namespace

std::unique_ptr<VisualizationHelper> VisualizationHelper::build(int w, int h, accelerated::Queue &q, const api::InternalAPI::DebugParameters &parameters) {
    return std::unique_ptr<VisualizationHelper>(new VisualizationHelperImplementation(w, h, q, parameters));
}

PoseOverlayHandler::~PoseOverlayHandler() = default;
TaggedFrameVisualizer::~TaggedFrameVisualizer() = default;

// Shorten pose histories to save memory. Cuts all the histories from same timestamp
// so that pose track visualizations remain comparable.
void trimPoseHistories(std::map<api::PoseHistory, std::vector<api::Pose>> &poseHistories) {
    if (poseHistories[api::PoseHistory::OUR].size() <= MAX_POSITION_HISTORY_LENGTH) {
        return;
    }
    assert(HISTORY_ERASE_BLOCK_SIZE < MAX_POSITION_HISTORY_LENGTH);
    double t = poseHistories[api::PoseHistory::OUR][HISTORY_ERASE_BLOCK_SIZE].time;

    poseHistories[api::PoseHistory::OUR].erase(
        poseHistories[api::PoseHistory::OUR].begin(),
        poseHistories[api::PoseHistory::OUR].begin() + HISTORY_ERASE_BLOCK_SIZE);
    for (auto it = poseHistories.begin(); it != poseHistories.end(); ++it) {
        if (it->first == api::PoseHistory::OUR) continue;
        int ind = odometry::views::getIndexWithTime(it->second, t);
        assert(ind >= 0);
        it->second.erase(
            it->second.begin(),
            it->second.begin() + ind);
    }
}

VisualizationHelper::~VisualizationHelper() = default;

}
