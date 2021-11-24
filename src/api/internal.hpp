#ifndef DAZZLING_API_INTERNAL_HPP
#define DAZZLING_API_INTERNAL_HPP

// Do not include this in algorithm code, use types.hpp instead

#include "vio.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"
#include <string>
#include <functional>

#include <nlohmann/json_fwd.hpp>

namespace api {
/**
 * Available pose overlay visualizations and their order.
 */
enum class PoseHistory {
    OUR = 0,
    GROUND_TRUTH = 1,
    ARKIT,
    OUR_PREVIOUS,
    GPS,
    RTK_GPS,
    EXTERNAL,
    ARENGINE,
    ARCORE,
    REALSENSE,
    ZED
};
}

using PoseHistoryMap = std::map<api::PoseHistory, std::vector<api::Pose>>;

// forward declarations
namespace slam { struct DebugAPI; class MapPointRecord; }
namespace odometry { class DebugAPI; struct TaggedFrame; }
namespace cv { class Mat; }
namespace api_visualization_helpers { struct VisualizationHelper; }

/**
 * Internal API for tracker debugging. Subject to change and not intended
 * for production use. OpenCV matrices may be used here, but preferably
 * as references so that the forward declaration mechanism works and the user
 * of this API is not compelled to include OpenCV headers
 *
 * This API is partially thread-safe:
 *  - addFrame / recordFrame and visualizeX methods must be called from the
 *    same thread
 *  - In case of GPU-backed images, it must be the OpenGL thread
 */
namespace api {
using Image = accelerated::Image;

class InternalAPI : public VioApi {
public:
    /**
     * Type of visualization to draw on color image.
     *
     * The difference between PROCESSED_VIDEO and PLAIN_VIDEO is that the former
     * delays its output until the odometry component has processed the frame.
     * This syncs it more precisely with pose output associated with the algorithm
     * output, possibly making AR experience better (as precise syncing seems more
     * important than minimal lag).
     */
    enum class VisualizationMode {
        NONE = 0,
        PLAIN_VIDEO = 1,
        TRACKER_ONLY = 2,
        TRACKS = 3,
        DEBUG_VISUALIZATION = 4,
        PROCESSED_VIDEO = 5,
        OPTICAL_FLOW = 6,
        OPTICAL_FLOW_FAILURES = 7,
        TRACKS_ALL = 8,
        CORNER_MEASURE = 10,
        STEREO_MATCHING = 11,
        STEREO_EPIPOLAR = 12,
        STEREO_DISPARITY = 13,
        STEREO_DEPTH = 14
    };

    struct Parameters {
        odometry::Parameters parameters;

        int inputFrameWidth;
        int inputFrameHeight;
    };

    struct DebugParameters {
        Parameters api;
        VisualizationMode visualization = VisualizationMode::DEBUG_VISUALIZATION;

        bool visualizePoseWindow = false; // used by commandline
        bool visualizePoseOverlay = false;
        bool visualizePointCloud = false;
        bool mobileVisualizations = false;

        /**
         * How many color images to buffer in memory for visualization purposes.
         * This mechanism is needed for debug visualizations since the last
         * processed frame lags behind the last input frame by (small) varying
         * number of frames. This parameter is the maximum supported lag. If the
         * lag is larger in some case, this does not crash but may display garbage.
         */
        unsigned nVisualBuffers = 10;

        /**
         * Path to non-existing file to which JSONL recording of the session
         * will be created. If empty, no recording takes place. Recording ends
         * and the file is closed when the API is deconstructed.
         */
        std::string recordingPath = "";
        /** If true, do not actually run the algorithm, just record data */
        bool recordingOnly = false;
        /** If false, do not record sensors or frame metadata */
        bool recordInputs = true;
        /**
         * If non-empty, record the video stream to this file in the monocular
         * case. In the stereo case, the other camera video file name(s) are
         * derived from this one by substituting e.g., /path/to/example.avi
         * -> /path/to/example2.avi.
         */
        std::string videoRecordingPath = "";
        /**
         * FPS metadata to be included in video recording. Does not affect the
         * image data, just informs how fast the video should be played.
         */
        float videoRecordingFps = 30;

        /**
         * If camera parameters are set relative to some other resolution than
         * the images input to the API, this is used to scale the parameters.
         */
        double videoAlgorithmScale = 1.0;
    };

    struct Output : public VioOutput {
        /** Focal length of this frame in pixels */
        float focalLength;

        /** Stationarity detection results */
        bool stationaryVisual;

        /** IMU biases */
        Vector3d meanBGA, meanBAA, meanBAT;
        /** IMU bias uncertainties */
        Vector3d covDiagBGA, covDiagBAA, covDiagBAT;

        /** Text debug information */
        std::string stateAsString;

        std::shared_ptr<odometry::TaggedFrame> taggedFrame;

        // note: questionable to copy this
        PoseHistoryMap poseHistories;

        // additional JSON data
        std::string additionalData;

        std::string asJson() const final;
    };

    ~InternalAPI();

    /**
     * Set secondFrame to nullptr to record only video.
     */
    virtual void recordFrames(
        double firstT, double secondT,
        double firstFocalLength, double secondFocalLength,
        Image *firstFrame = nullptr, Image *secondFrame = nullptr
    ) = 0;

    virtual void recordFrames(
        double firstT, double secondT,
        api::CameraParameters *firstIntrinsic = nullptr, api::CameraParameters *secondIntrinsic = nullptr,
        Image *firstFrame = nullptr, Image *secondFrame = nullptr
    ) = 0;

    /**
     * Add gyro sample to sample sync. Use this together with `processSample()`
     * and processAll=false for access to odometry output on each individual
     * synced sample.
     *
     * @param t Sample timestamp.
     * @param sample Sample vector.
     * @param processAll Iff true, process all new output from sample sync.
     */
    virtual void addGyroInternal(double t, const Vector3d &sample, bool processAll) = 0;
    /**
     * Process a single sample from sample sync if available.
     *
     * @return True if a sample was processed
     */
    virtual bool processSample() = 0;

    // The data must be in JSON format. (optional feature): certain fields could
    // have special meanings, for example "time" in the root could override the
    // timestamp, which all JSONL fields in the output log have
    virtual void addAuxiliaryJsonData(const std::string &auxiliaryJsonData) = 0;

    // per-frame varying camera parameters (e.g. Android / iOS)
    virtual void addFrameMonoVarying(
        double t, const CameraParameters &cam,
        int w, int h, const std::uint8_t *data,
        ColorFormat colorFormat, int tag = 0) = 0;

    // OpenGL GPU extension. Cannot be used simultaneously with CPU-based
    // "addFrameX" variants (aborts if attempted).

    // If set, called whenever there are new items that should be processed
    // in the OpenGL thread by calling "processOpenGl". Here "OpenGL thread"
    // means any thread which has an OpenGl context, which is active  when
    // these methods are used.
    std::function<void()> onOpenGlWork;

    // These methods must be called from the OpenGL thread. The texture ID
    // must be valid in this OpenGL context
    virtual void addFrameMonoOpenGl(
        double t, const CameraParameters &cam,
        int w, int h, int externalOesTextureId,
        ColorFormat colorFormat, int tag = 0) = 0;

    virtual void addFrameMonoOpenGl(
        double t, int w, int h,
        int externalOesTextureId,
        ColorFormat colorFormat,
        int tag = 0) = 0;

    virtual void addFrameStereoOpenGl(
        double t, int w, int h,
        int externalOesTextureId0, int externalOesTextureId1,
        ColorFormat colorFormat,
        int tag = 0) = 0;

    // Process pending OpenGL operations, if any. If OpenGL inputs are used,
    // must be called
    virtual void processOpenGl() = 0;

    // Must be called from the OpenGL thread and return before the main
    // destructor is called (if OpenGL inputs have been used). All created
    // "visualizations" must also be deconstructed before calling this
    virtual void destroyOpenGl() = 0;

    virtual void lockBiases() = 0;
    virtual void conditionOnLastPose() = 0;

    virtual CameraParameters fallbackIntrinsic(
        CameraParameters intrinsicPerFrame,
        int width,
        int height,
        bool secondCamera
    ) const = 0;

    virtual void setPoseHistory(PoseHistory kind, const std::vector<Pose> &poseHistory) = 0;
    virtual bool getPoseOverlayHistoryExists(PoseHistory poseHistory) const = 0;
    virtual void setPoseOverlayHistoryShown(PoseHistory poseHistory, bool value) = 0;
    virtual bool getPoseOverlayHistoryShown(PoseHistory poseHistory) const = 0;

    virtual void connectDebugApi(odometry::DebugAPI &debugApi) = 0;

    /**
     * Process all queued GPU / visual operations. Must be called
     * on the GL thread. This also happens on addFrame, but using this may
     * increase throughput or even be mandatory with certain (experimental)
     * parameter combinations (see above)
     */
    // virtual void processOpenGl() = 0;

    // extra visualizations
    virtual void visualizeKfCorrelation(cv::Mat &target) = 0;
    virtual void visualizeCovarianceMagnitudes(cv::Mat &target) = 0;

    virtual std::vector<Pose> getPoseHistory(PoseHistory kind) const = 0;
    virtual void appendPoseHistoryARKit(Pose) = 0;
    virtual void appendPoseHistoryARCore(double t, Pose) = 0;
    virtual void appendPoseHistoryAREngine(double t, Pose) = 0;
    virtual void appendPoseHistoryGps(
            double t,
            double latitude,
            double longitude,
            double horizontalUncertainty,
            double altitude) = 0;

    virtual void setParameterString(std::string parameterString) = 0;
    virtual void setVisualization(VisualizationMode visualization) = 0;
    virtual void setPoseOverlay(bool enabled) = 0;
    virtual void recordJsonString(const std::string &line) = 0;
    virtual void recordJson(const nlohmann::json &j) = 0;
    virtual api_visualization_helpers::VisualizationHelper* getVisualizationHelper() = 0;
};

std::unique_ptr<api::InternalAPI> buildVio(const InternalAPI::DebugParameters &parameters);
// std::unique_ptr<VioApi> buildVio(const InternalAPI::Parameters &parameters);

} // namespac api

#endif
