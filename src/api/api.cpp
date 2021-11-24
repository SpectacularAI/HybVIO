#include <array>
#include <map>
#include <mutex>

#include "../util/util.hpp"
#include "internal.hpp"
#include "jsonl-recorder/recorder.hpp"
#include "../odometry/debug.hpp"
#include "../odometry/control.hpp"
#include "../odometry/tagged_frame.hpp"
#include "../odometry/ekf.hpp"
#include "../views/api_visualization_helpers.hpp"
#include "../views/visualization_internals.hpp"
#include "../views/views.hpp"
#include "../api/type_convert.hpp"
#include "../util/gps.hpp"
#include "../util/allocator.hpp"
#include "../util/bounded_processing_queue.hpp"
#include "../tracker/camera.hpp"
#include "../tracker/image.hpp"
#include "output_buffer.hpp"
#include "vio.hpp"
#include "visualizations.hpp"
#include <nlohmann/json.hpp>

#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/cpu/image.hpp>
#include <accelerated-arrays/cpu/operations.hpp>
#ifdef DAZZLING_GPU_ENABLED
#include <accelerated-arrays/opengl/image.hpp>
#include <accelerated-arrays/opengl/operations.hpp>
#endif
#include <accelerated-arrays/opencv_adapter.hpp>

// random utilities only used for this class
#include "implementation_helpers.hpp"

#include "../commandline/parameters.hpp"

using json = nlohmann::json;

namespace api {

InternalAPI::~InternalAPI() = default;

// TODO: move to accelerated-arrays
struct NotifyingQueue : public accelerated::Queue {
private:
    std::unique_ptr<accelerated::Queue> queue;
    std::function<void()> notifyHook;

public:
    NotifyingQueue(const std::function<void()> &hook)
    : queue(accelerated::Processor::createQueue()), notifyHook(hook) {}

    void processAll() final { queue->processAll(); }
    bool processOne() final { return queue->processOne(); }

     accelerated::Future enqueue(const std::function<void()> &op) final {
        auto fut = queue->enqueue(op);
        // TODO: check if empty for effciency
        notifyHook();
        return fut;
    }
};

class InternalAPIImplementation : public InternalAPI {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    InternalAPIImplementation(const DebugParameters &originalParameters)
        :
        parameters(applyAutoParameters(originalParameters)),
        imageProcessingQueue(std::make_unique<NotifyingQueue>([this]() {
            if (onOpenGlWork) onOpenGlWork();
        })),
        outputStore(std::make_unique<util::Allocator<Output>>(
            []() { return std::make_unique<Output>(); }
        )),
        control(parameters.recordingOnly ? nullptr :
            odometry::Control::build(parameters.api.parameters)),
        controlProcessingQueue(parameters.api.parameters.odometry.processingQueueSize)
    {

        if (control) {
            log_debug("New tracking session, input resolution %d x %d",
                      parameters.api.inputFrameWidth,
                      parameters.api.inputFrameHeight);
            log_debug("Focal length is set to %g (or %g, %g), principal point (%g, %g)",
                      parameters.api.parameters.tracker.focalLength,
                      parameters.api.parameters.tracker.focalLengthX,
                      parameters.api.parameters.tracker.focalLengthY,
                      parameters.api.parameters.tracker.principalPointX,
                      parameters.api.parameters.tracker.principalPointY);
        }

        if (!parameters.recordingPath.empty()) {
            log_info("recoding to %s", parameters.recordingPath.c_str());
            recorder = recorder::Recorder::build(parameters.recordingPath, parameters.videoRecordingPath);
            recorder->setVideoRecordingFps(parameters.videoRecordingFps);
        }
    }

    void initVisu(int width, int height) {
        visuData = api_visualization_helpers::VisualizationHelper::build(
            width,
            height,
            *imageProcessingQueue,
            parameters);
    }

    // VioApi

    void addAcc(double t, const api::Vector3d &sample) final {
        if (recorder && parameters.recordInputs) {
            recorder->addAccelerometer(t, sample.x, sample.y, sample.z);
        }

        if (parameters.recordingOnly) return;

        //log_debug("acc  %ld(%g): %g", timeNanos, t, val.x);
        assert(control && "input after GL cleanup");
        control->processAccelerometerSample(t, sample);
    }

    void addGyro(double t, const api::Vector3d &sample) final {
        addGyroInternal(t, sample, true);
    }

    void addAuxiliaryJsonData(const std::string &auxiliaryJsonData) final {
        json j = json::parse(auxiliaryJsonData);
        bool hasTime = j.find("time") != j.end();
        bool hasGps = j.find("gps") != j.end();
        if (hasTime && hasGps) {
            appendPoseHistoryGps(
                j["time"].get<double>(),
                j["gps"]["latitude"].get<double>(),
                j["gps"]["longitude"].get<double>(),
                j["gps"]["accuracy"].get<double>(),
                j["gps"]["altitude"].get<double>());
        } else if (recorder) {
            recorder->addJsonString(auxiliaryJsonData);
        }
    };

    void addFrameMono(
        double t,
        int w, int h, const std::uint8_t *data,
        VioApi::ColorFormat colorFormat, int tag) final
    {
        CameraParameters emptyParams;
        addFrameMonoVarying(t, emptyParams, w, h, data, colorFormat, tag);
    }

    void addFrameMonoVarying(
        double t,
        const CameraParameters &cam,
        int w, int h, const std::uint8_t *data,
        VioApi::ColorFormat colorFormat, int tag) final
    {
        int channels = numChannels(colorFormat);
        std::unique_ptr<Image> inputImage = accelerated::cpu::Image::createReference(
            w, h, channels, accelerated::cpu::Image::DataType::UFIXED8,
            const_cast<std::uint8_t*>(data));
        addFrame(t, *inputImage, cam, tag);
    }

    void addFrameStereo(
        double t,
        int w, int h, const std::uint8_t *data0, const std::uint8_t *data1,
        VioApi::ColorFormat colorFormat, int tag) final
    {
        int channels = numChannels(colorFormat);
        CameraParameters emptyParams;
        std::unique_ptr<Image> inputImage0 = accelerated::cpu::Image::createReference(
            w, h, channels, accelerated::cpu::Image::DataType::UFIXED8,
            const_cast<std::uint8_t*>(data0));
        std::unique_ptr<Image> inputImage1 = accelerated::cpu::Image::createReference(
            w, h, channels, accelerated::cpu::Image::DataType::UFIXED8,
            const_cast<std::uint8_t*>(data1));
        addStereoFrames(t, t, *inputImage0, *inputImage1, emptyParams, emptyParams, tag);
    }

    void addFrameMonoOpenGl(
        double t,
        int w, int h, int externalOesTextureId,
        ColorFormat colorFormat, int tag) final
    {
        CameraParameters emptyParams;
        addFrameMonoOpenGl(t, emptyParams, w, h, externalOesTextureId, colorFormat, tag);
    }

    void addFrameMonoOpenGl(
        double t, const CameraParameters &cam,
        int w, int h, int externalOesTextureId,
        ColorFormat colorFormat, int tag) final
    {
        std::unique_ptr<Image> inputImage = wrapTexture(w, h, externalOesTextureId, colorFormat);
        addFrame(t, *inputImage, cam, tag);
    }

    void addFrameStereoOpenGl(
        double t,
        int w, int h, int externalOesTextureId0, int externalOesTextureId1,
        ColorFormat colorFormat, int tag) final
    {
        CameraParameters emptyParams;
        std::unique_ptr<Image> inputImage0 = wrapTexture(w, h, externalOesTextureId0, colorFormat);
        std::unique_ptr<Image> inputImage1 = wrapTexture(w, h, externalOesTextureId1, colorFormat);
        addStereoFrames(t, t, *inputImage0, *inputImage1, emptyParams, emptyParams, tag);
    };

    void processOpenGl() final {
        if (imageProcessingQueue) imageProcessingQueue->processAll();
    };

    virtual void destroyOpenGl() {
        onOpenGlWork = {};

        // clean up accelerated arrays stuff in GL thread
        {
            std::lock_guard<std::mutex> lock(outputMutex);
            outputStore.reset();
        }

        controlProcessingQueue.enqueueAndWait([this]() {
            control.reset();
        });

        visuData.reset();
        colorFrameBuffer.reset();
        trackerImageFactory.reset();
        copyAsColorFrameOp = {};
        accOpsFactory.reset();
        imageProcessingQueue->processAll();
        accImageFactory.reset();
        imageProcessingQueue->processAll();
        imageProcessingQueue.reset();
    };

    // This could also be detached from the main API and put into its own file
    // so that visualization can only access Output but not any algorithm
    // internals. This is a bit of a double-edged sword. For efficiency, it
    // would be good if some debug data was only populated if requested: for
    // example, we could store the "color frame" in debug data only if some
    // color visualization is activated. This could also be done using algorithm
    // parameters. This also enables other hacks such as directly pushing data
    // from algorithm internals (e.g., SLAM data) into the "Visualization"
    // object without using DebugData
    std::shared_ptr<Visualization> createVisualization(const std::string &type) final {
        if (type == "VIDEO") {
            enableVisualization = true;
            return std::make_shared<VisualizationVideoOutput>(this);
        } else if (type == "KF_CORRELATION") {
            return std::make_shared<VisualizationKfCorrelation>(this);
        } else if (type == "POSE") {
            enableVisualization = true;
            return std::make_shared<VisualizationPose>(this);
        } else if (type == "COVARIANCE_MAGNITUDES") {
            return std::make_shared<VisualizationCovarianceMagnitudes>(this);
        }
        assert(false && "Unsupported visualization type!");
    }

    void connectDebugApi(odometry::DebugAPI &debugApi) final {
        controlProcessingQueue.enqueue([this, &debugApi](){
            assert(control);
            control->connectDebugAPI(debugApi);
        });
    };

    bool getPoseOverlayHistoryExists(PoseHistory poseHistory) const final {
        assert(visuData);
        return visuData->poseOverlayHistory().getExists(poseHistory);
    }

    void setPoseOverlayHistoryShown(PoseHistory poseHistory, bool value) final {
        assert(visuData);
        visuData->poseOverlayHistory().setShown(poseHistory, value);
    }

    bool getPoseOverlayHistoryShown(PoseHistory poseHistory) const final {
        assert(visuData);
        return visuData->poseOverlayHistory().getShown(poseHistory);
    }

    void setPoseHistory(PoseHistory kind, const std::vector<api::Pose> &poseHistory) final {
        std::lock_guard<std::mutex> lock(outputMutex);
        std::vector<api::Pose> copy;

        switch (kind) {
            case PoseHistory::ARCORE:
            case PoseHistory::ARENGINE:
            case PoseHistory::ZED:
                copy = poseHistory;
                for (Pose &pose : copy) {
                    convertViotesterAndroidPose(pose, parameters.api.parameters.imuToCamera);
                }
                poseHistories[kind] = std::move(copy);
                break;
            default:
                poseHistories[kind] = poseHistory;
                break;
        }
    }

    void lockBiases() final {
        controlProcessingQueue.enqueue([this]() {
            assert(control);
            control->lockBiases();
        });
    }

    void conditionOnLastPose() final {
        controlProcessingQueue.enqueue([this]() {
            assert(control);
            control->conditionOnLastPose();
        });
    }

    CameraParameters fallbackIntrinsic(
        CameraParameters intrinsicPerFrame,
        int width,
        int height,
        bool secondCamera
    ) const {
        // It is important this is an instance method that doesn't take the `parameters`
        // as argument. We want to use this internal parameters struct field that has
        // automatically set focalLengthX etc. values, in order to make them available
        // outside the API.
        const auto &pt = parameters.api.parameters.tracker;

        double autoPx = 0.5 * static_cast<double>(width);
        double autoPy = 0.5 * static_cast<double>(height);
        CameraParameters intrinsic;
        if (secondCamera) {
            intrinsic.focalLengthX = setIntrinsic("fx 2",
                intrinsicPerFrame.focalLengthX, pt.secondFocalLengthX);
            intrinsic.focalLengthY = setIntrinsic("fy 2",
                intrinsicPerFrame.focalLengthY, pt.secondFocalLengthY);
            intrinsic.principalPointX = setIntrinsic("px 2",
                intrinsicPerFrame.principalPointX, pt.secondPrincipalPointX, autoPx);
            intrinsic.principalPointY = setIntrinsic("py 2",
                intrinsicPerFrame.principalPointY, pt.secondPrincipalPointY, autoPy);
        }
        else {
            intrinsic.focalLengthX = setIntrinsic("fx 1",
                intrinsicPerFrame.focalLengthX, pt.focalLengthX);
            intrinsic.focalLengthY = setIntrinsic("fy 1",
                intrinsicPerFrame.focalLengthY, pt.focalLengthY);
            intrinsic.principalPointX = setIntrinsic("px 1",
                intrinsicPerFrame.principalPointX, pt.principalPointX, autoPx);
            intrinsic.principalPointY = setIntrinsic("py 1",
                intrinsicPerFrame.principalPointY, pt.principalPointY, autoPy);
        }
        return intrinsic;
    }

    // InternalAPI

    int numChannels(ColorFormat colorFormat) {
        if (colorFormat == VioApi::ColorFormat::GRAY)
            return 1;
        return colorFormat == VioApi::ColorFormat::RGB ? 3 : 4;
    }

    std::unique_ptr<Image> wrapTexture(int w, int h, int externalOesTextureId, ColorFormat colorFormat) {
        #ifdef DAZZLING_GPU_ENABLED
        if (colorFormat == VioApi::ColorFormat::RGBA_EXTERNAL_OES)
            ensureFactories(Image::StorageType::GPU_OPENGL_EXTERNAL);
        else
            ensureFactories(Image::StorageType::GPU_OPENGL);
        auto factory = dynamic_cast<accelerated::opengl::Image::Factory*>(accImageFactory.get());
        using Ufixed8 = accelerated::FixedPoint<std::uint8_t>;
        if (colorFormat == VioApi::ColorFormat::GRAY)
            return factory->wrapTexture<Ufixed8, 1>(externalOesTextureId, w, h,
                accelerated::ImageTypeSpec::StorageType::GPU_OPENGL);
        if (colorFormat == VioApi::ColorFormat::RGB)
            return factory->wrapTexture<Ufixed8, 3>(externalOesTextureId, w, h,
                accelerated::ImageTypeSpec::StorageType::GPU_OPENGL);
        if (colorFormat == VioApi::ColorFormat::RGBA)
            return factory->wrapTexture<Ufixed8, 4>(externalOesTextureId, w, h,
                accelerated::ImageTypeSpec::StorageType::GPU_OPENGL);
        if (colorFormat == VioApi::ColorFormat::RGBA_EXTERNAL_OES)
            return factory->wrapTexture<Ufixed8, 4>(externalOesTextureId, w, h,
                accelerated::ImageTypeSpec::StorageType::GPU_OPENGL_EXTERNAL);
        #endif
        assert(false && "Unsupported ColorFormat!");
    }

    void recordFrames(
        double firstT, double secondT,
        double firstFocalLength, double secondFocalLength,
        Image *firstFrame, Image *secondFrame
    ) final {
        api::CameraParameters firstIntrinsic(firstFocalLength);
        api::CameraParameters secondIntrinsic(secondFocalLength);
        recordFrames(
            firstT, secondT,
            &firstIntrinsic, &secondIntrinsic,
            firstFrame, secondFrame);
    }

    void recordFrames(
        double firstT, double secondT,
        api::CameraParameters *firstIntrinsic, api::CameraParameters *secondIntrinsic,
        Image *firstFrame, Image *secondFrame
    ) final {
        recordFramesInternal(
            firstT, secondT,
            firstIntrinsic, secondIntrinsic,
            firstFrame, secondFrame);
    }

    void addGyroInternal(double t, const Vector3d &sample, bool processAll) final {
        if (recorder && parameters.recordInputs) {
            recorder->addGyroscope(t, sample.x, sample.y, sample.z);
        }

        if (parameters.recordingOnly) return;

        assert(control && "input after GL cleanup");
        control->processGyroSample(t, sample);
        if (processAll) {
            controlProcessingQueue.enqueue([this]() {
                constexpr size_t MAX_SAMPLES = 2; // TODO: 1 could be enough
                processSampleInternal(MAX_SAMPLES, true);
            });
        }
    }

    bool processSample() final {
        // processSample cannot be used together with the processing queue
        assert(controlProcessingQueue.maxSize() == 0);
        return processSampleInternal(1, false);
    }

    std::vector<api::Pose> getPoseHistory(PoseHistory kind) const final {
        std::lock_guard<std::mutex> lock(const_cast<InternalAPIImplementation&>(*this).outputMutex);
        auto search = poseHistories.find(kind);
        if (search != poseHistories.end()) {
            return search->second;
        }
        return {};
    }

    void appendPoseHistoryARKit(Pose pose) final {
        if (recorder) {
            recorder->addARKit(pose);
        }
        std::lock_guard<std::mutex> lock(outputMutex);
        poseHistories[PoseHistory::ARKIT].push_back(pose);
    }

    void appendPoseHistoryARCore(double t, Pose pose) final {
        (void)t;
        convertViotesterAndroidPose(pose, parameters.api.parameters.imuToCamera);
        {
            std::lock_guard<std::mutex> lock(outputMutex);
            poseHistories[PoseHistory::ARCORE].push_back(pose);
        }
    }

    void appendPoseHistoryAREngine(double t, Pose pose) final {
        (void)t;
        convertViotesterAndroidPose(pose, parameters.api.parameters.imuToCamera);
        {
            std::lock_guard<std::mutex> lock(outputMutex);
            poseHistories[PoseHistory::ARENGINE].push_back(pose);
        }
    }

    void appendPoseHistoryGps(
            double t,
            double latitude,
            double longitude,
            double horizontalUncertainty,
            double altitude
    ) final {
        if (recorder) {
            recorder->addGps(t, latitude, longitude, horizontalUncertainty, altitude);
        }
        std::lock_guard<std::mutex> lock(outputMutex);
        poseHistories[PoseHistory::GPS].push_back(Pose {
            .time = t,
            .position = api::eigenToVector(gpsToLocal.convert(latitude, longitude, altitude, horizontalUncertainty)),
            .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 }
        });
    }

    void setParameterString(std::string parameterString) final {
        controlProcessingQueue.enqueueAndWait([this, parameterString]() {
            auto str = std::istringstream(parameterString);
            odometry::setParameterString(parameters.api.parameters, str);
        });
    }

    void setVisualization(VisualizationMode visualization) final {
        parameters.visualization = visualization;
    }

    void setPoseOverlay(bool enabled) final {
        parameters.visualizePoseOverlay = enabled;
    }

    void visualizeKfCorrelation(cv::Mat &target) final {
        visualizeCovariance(target, true);
    }

    void visualizeCovarianceMagnitudes(cv::Mat &target) final {
        visualizeCovariance(target, false);
    }

    void recordJsonString(const std::string &line) final {
        if (recorder) {
            recorder->addJsonString(line);
        }
    }

    void recordJson(const nlohmann::json &j) final {
        if (recorder) {
            recorder->addJson(j);
        }
    }

private:
    // helper methods
    void addFrame(double t, Image &inputImage, api::CameraParameters intrinsicPerFrame, int tag) {
        if (!imageProcessingQueue) {
            log_warn("ignoring frame after visual destroy");
            return;
        }
        assert(tag >= 0);

        // If non-empty, colorFrame is an RGBA copy of inputImage, managed in
        // a buffer in this class
        std::shared_ptr<Image> colorFrame = recordFramesInternal(
            t, 0.0,
            &intrinsicPerFrame, nullptr,
            &inputImage, nullptr)[0];

        if (parameters.recordingOnly) return;
        assert(control);

        std::unique_ptr<odometry::TaggedFrame> taggedFrame;

        const auto &pt = parameters.api.parameters.tracker;
        auto cameraKind = pt.fisheyeCamera ? tracker::Camera::Kind::FISHEYE : tracker::Camera::Kind::PINHOLE;
        auto intrinsic = fallbackIntrinsic(intrinsicPerFrame, inputImage.width, inputImage.height, false);
        auto camera = buildCamera(intrinsic, cameraKind, pt,
            inputImage.width, inputImage.height, pt.distortionCoeffs);
        // This copies the, possibly colored input frame to a gray buffer/texture
        auto grayImage = copyAsGrayFrame(inputImage, std::move(camera));

        if (enableVisualization) {
            // reuse the color frame already stored by recording if possible
            // but always show rectified output if rectifier is used. Otherwise
            // the tracked features and other stuff appears in a wrong place
            if (pt.useRectification) colorFrame = copyAsColorFrame(grayImage->getAccImage());
            else if (!colorFrame) colorFrame = copyAsColorFrame(inputImage);
            if (!visuData) initVisu(inputImage.width, inputImage.height);
            taggedFrame = visuData->createTaggedFrame(tag, colorFrame);
        }

        control->processFrame(t, std::move(grayImage), std::move(taggedFrame));
    }

    void addStereoFrames(
        double firstT, double secondT,
        Image &firstImage, Image &secondImage,
        api::CameraParameters firstIntrinsicPerFrame,
        api::CameraParameters secondIntrinsicPerFrame,
        int tag)
    {
        if (!imageProcessingQueue) {
            log_warn("ignoring stereo frame after visual destroy");
            return;
        }

        auto colorFrames = recordFramesInternal(
            firstT, secondT,
            &firstIntrinsicPerFrame, &secondIntrinsicPerFrame,
            &firstImage, &secondImage);

        if (parameters.recordingOnly) return;

        const auto &pa = parameters.api.parameters;
        const auto &pt = pa.tracker;
        auto cameraKind = pt.fisheyeCamera ? tracker::Camera::Kind::FISHEYE : tracker::Camera::Kind::PINHOLE;
        auto firstIntrinsic = fallbackIntrinsic(firstIntrinsicPerFrame,
            firstImage.width, firstImage.height, false);
        auto firstCamera = buildCamera(firstIntrinsic, cameraKind, pt,
            firstImage.width, firstImage.height, pt.distortionCoeffs);
        auto secondIntrinsic = fallbackIntrinsic(secondIntrinsicPerFrame,
            secondImage.width, secondImage.height, true);
        auto secondCamera = buildCamera(secondIntrinsic, cameraKind, pt,
            secondImage.width, secondImage.height, pt.secondDistortionCoeffs);

        // "copy as stereo gray frame"
        ensureFactories(firstImage.storageType);

        auto stereoFrames = trackerImageFactory->buildStereo(
            firstImage, secondImage,
            std::move(firstCamera),
            std::move(secondCamera));

        std::unique_ptr<odometry::TaggedFrame> taggedFrame;
        if (enableVisualization) {
            if (pt.useRectification) {
                colorFrames[0] = copyAsColorFrame(stereoFrames.first->getAccImage());
                colorFrames[1] = copyAsColorFrame(stereoFrames.second->getAccImage());
            }
            else {
                if (!colorFrames[0]) colorFrames[0] = copyAsColorFrame(firstImage);
                if (!colorFrames[1]) colorFrames[1] = copyAsColorFrame(secondImage);
            }
            if (!visuData) initVisu(firstImage.width, firstImage.height);
            taggedFrame = visuData->createTaggedFrameStereo(tag,
                colorFrames[0], colorFrames[1]);
        }

        imageProcessingQueue->processAll();

        control->processStereoFrames(firstT,
            std::move(stereoFrames.first),
            std::move(stereoFrames.second),
            std::move(taggedFrame));
    }

    // Internal because returns Image data.
    std::array< std::shared_ptr<Image >, 2 > recordFramesInternal(
        double firstT, double secondT,
        api::CameraParameters *firstIntrinsic, api::CameraParameters *secondIntrinsic,
        Image *firstFrame, Image *secondFrame) {
        // Some care needs to be taken here to handle the input images in an
        // optimal manner: Image may be a reference to data that is as close to
        // the camera and as hostile towards direct pixel access as possible on
        // an Android device: an GL_TEXTURE_EXTERNAL_OES surface. To read it,
        // it must first be transferred to a normal GPU texture and then, if
        // necessary, to normal CPU-accessible RAM.

        std::array< std::shared_ptr<Image >, 2 > colorFrames;
        if (recorder) {
            if (parameters.recordInputs) {
                // this part copies to a normal OpenGL texture. For CPU images
                // it's just memcpy. In any case, the copy image is now managed by
                // the allocator buffer in this object. To avoid doing this twice,
                // the same texture is also used for visualizations (see below)
                if (firstFrame != nullptr) {
                    colorFrames[0] = copyAsColorFrame(*firstFrame);
                }
                if (secondFrame != nullptr) {
                    colorFrames[1] = copyAsColorFrame(*secondFrame);
                }
                recordFramesDirectlyReadable(firstT, secondT,
                    firstIntrinsic, secondIntrinsic,
                    colorFrames[0].get(), colorFrames[1].get());
            }
        }
        return colorFrames;
    }

    void recordFramesDirectlyReadable(
        double firstT, double secondT,
        api::CameraParameters *firstIntrinsic, api::CameraParameters *secondIntrinsic,
        Image *firstFrame, Image *secondFrame)
    {
        assert(recorder && parameters.recordInputs);
        int cameraCount = 1;
        if (secondFrame) {
            cameraCount = 2;
        }
        cv::Mat firstCvMat; // TODO: Use recorder->getEmptyFrames(...)
        if (firstFrame != nullptr) {
            firstCvMat = getAsCvMat(*firstFrame);
        }
        assert(firstIntrinsic);
        auto frameData0 = recorder::FrameData {
            .t = firstT,
            .cameraInd = 0,
            .focalLengthX = firstIntrinsic->focalLengthX,
            .focalLengthY = firstIntrinsic->focalLengthY,
            .px = firstIntrinsic->principalPointX,
            .py = firstIntrinsic->principalPointY,
            .frameData = nullptr
        };
        if (cameraCount == 2) {
            cv::Mat secondCvMat; // TODO: Use recorder->getEmptyFrames(...)
            if (secondFrame != nullptr) {
                assert(firstFrame != nullptr);
                secondCvMat = getAsCvMat(*secondFrame);
            }
            assert(secondIntrinsic);
            auto frameData1 = recorder::FrameData {
                .t = secondT,
                .cameraInd = 1,
                .focalLengthX = secondIntrinsic->focalLengthX,
                .focalLengthY = secondIntrinsic->focalLengthY,
                .px = secondIntrinsic->principalPointX,
                .py = secondIntrinsic->principalPointY,
                .frameData = nullptr
            };
            frameData0.frameData = firstFrame == nullptr ? nullptr : &firstCvMat;
            frameData1.frameData = secondFrame == nullptr ? nullptr : &secondCvMat;
            recorder->addFrameGroup(firstT, { frameData0, frameData1 });
        } else {
            frameData0.frameData = firstFrame == nullptr ? nullptr : &firstCvMat;
            recorder->addFrame(frameData0);
        }
    }

    nlohmann::json eigenToJson(const Eigen::Vector3d &vec) {
        nlohmann::json j = {{"x", vec[0]}, {"y", vec[1]}, {"z", vec[2]}};
        return j;
    }

    nlohmann::json eigenToJson(const Eigen::Matrix3d &cov) {
        nlohmann::json j = nlohmann::json::array({
            { cov(0, 0), cov(0, 1), cov(0, 2) },
            { cov(1, 0), cov(1, 1), cov(1, 2) },
            { cov(2, 0), cov(2, 1), cov(2, 2) }
        });
        return j;
    }

    api::Pose convertOutputPose(double t, const Eigen::Vector3d &odometryPosition, const Eigen::Vector4d &odometryOrientation) const {
        Eigen::Vector3d pos;
        Eigen::Vector4d ori;

        if (parameters.api.parameters.imuToOutput != Eigen::Matrix4d::Identity()) {
            pos = odometryPosition;
            ori = odometryOrientation;
        } else {
            const Eigen::Matrix4d worldToOutput = odometry::util::toWorldToCamera(
                odometryPosition,
                odometryOrientation,
                parameters.api.parameters.imuToOutput);
            odometry::util::toOdometryPose(worldToOutput, pos, ori, Eigen::Matrix4d::Identity());
            // Should also transform other parts of the state, like velocity.
        }
        return api::eigenToPose(t, pos, ori);
    }

    bool processSampleInternal(size_t maxSamples, bool allFrames) {
        if (!control) return false; // could happen in teardown

        bool anyProgress = true, curProgress, wasFrame = false;
        for (size_t i = 0; i < maxSamples || (wasFrame && allFrames); ++i) {
            auto progress = control->processSyncedSamples(1);
            wasFrame = progress == odometry::Control::SampleProcessResult::FRAMES;
            if (wasFrame) getOutputIfAvailable();
            curProgress = progress != odometry::Control::SampleProcessResult::NONE;
            if (curProgress) anyProgress = true;
            else break;
        }
        return anyProgress;
    }

    void getOutputIfAvailable() {
        std::unique_lock<std::mutex> outputLock(outputMutex);
        if (!outputStore) return; // after cleanupOpenGl
        auto nextOutput = outputStore->next();
        outputLock.unlock();

        if (!control) return;
        auto controlOutput = control->getOutput();
        nextOutput->tag = 0;
        nextOutput->taggedFrame = controlOutput.taggedFrame;
        if (nextOutput->taggedFrame) nextOutput->tag = nextOutput->taggedFrame->tag;

        nextOutput->stateAsString = control->stateAsString();
        nextOutput->poseTrail.clear();
        {
            nextOutput->poseTrail.reserve(controlOutput.poseTrailLength());
            for (size_t i = 0; i < controlOutput.poseTrailLength(); ++i) {
                nextOutput->poseTrail.push_back(
                    convertOutputPose(
                        controlOutput.poseTrailTimeStamp(i),
                        controlOutput.poseTrailPosition(i),
                        controlOutput.poseTrailOrientation(i)));
            }
        }

        {
            auto tmpPointCloud = controlOutput.pointCloud;

            nextOutput->pointCloud.clear();
            for (const auto &p : *tmpPointCloud) {
                nextOutput->pointCloud.push_back({
                    .id = p.id,
                    .position = api::eigenToVector(p.point),
                    .status = static_cast<int>(p.status)
                });
            }
        }

        nextOutput->pose = convertOutputPose(controlOutput.t, controlOutput.position(), controlOutput.orientation());
        nextOutput->status = controlOutput.trackingStatus;
        nextOutput->velocity = api::eigenToVector(controlOutput.velocity());
        nextOutput->focalLength = controlOutput.focalLength;
        nextOutput->positionCovariance = api::eigenToMatrix(controlOutput.positionCovariance());
        nextOutput->meanBGA = api::eigenToVector(controlOutput.meanBGA());
        nextOutput->meanBAA = api::eigenToVector(controlOutput.meanBAA());
        nextOutput->meanBAT = api::eigenToVector(controlOutput.meanBAT());
        nextOutput->covDiagBGA = api::eigenToVector(controlOutput.covDiagBGA());
        nextOutput->covDiagBAA = api::eigenToVector(controlOutput.covDiagBAA());
        nextOutput->covDiagBAT = api::eigenToVector(controlOutput.covDiagBAT());
        nextOutput->stationaryVisual = controlOutput.stationaryVisual;

        outputLock.lock();
        poseHistories[PoseHistory::OUR].push_back(nextOutput->pose);
        api_visualization_helpers::trimPoseHistories(poseHistories);
        // TODO: questionable. Note that it's also possible to just append
        // the changes to this reused object
        nextOutput->poseHistories = poseHistories;

        if (parameters.api.parameters.odometry.outputJsonExtras) {
            additionalOutputInfo["positionCovariance"] = eigenToJson(controlOutput.positionCovariance());
            additionalOutputInfo["velocityCovariance"] = eigenToJson(controlOutput.velocityCovariance());
            additionalOutputInfo["focalLength"] = controlOutput.focalLength;
            additionalOutputInfo["biasMean"]["gyroscopeAdditive"]
                = eigenToJson(controlOutput.meanBGA());
            additionalOutputInfo["biasMean"]["accelerometerAdditive"]
                = eigenToJson(controlOutput.meanBAA());
            additionalOutputInfo["biasMean"]["accelerometerTransform"]
                = eigenToJson(controlOutput.meanBAT());
            additionalOutputInfo["biasCovarianceDiagonal"]["gyroscopeAdditive"]
                = eigenToJson(controlOutput.covDiagBGA());
            additionalOutputInfo["biasCovarianceDiagonal"]["accelerometerAdditive"]
                = eigenToJson(controlOutput.covDiagBAA());
            additionalOutputInfo["biasCovarianceDiagonal"]["accelerometerTransform"]
                = eigenToJson(controlOutput.covDiagBAT());
            additionalOutputInfo["stationaryVisual"] = controlOutput.stationaryVisual;
            if (parameters.api.parameters.odometry.outputJsonPoseTrail) {
                auto &jsonTrail = additionalOutputInfo["poseTrail"];
                jsonTrail.clear();
                for (size_t i = 0; i < controlOutput.poseTrailLength(); ++i) {
                    const auto apiPose = convertOutputPose(
                        controlOutput.poseTrailTimeStamp(i),
                        controlOutput.poseTrailPosition(i),
                        controlOutput.poseTrailOrientation(i));
                    nlohmann::json jsonPose = {
                        {"time", apiPose.time},
                        {"position", {
                            {"x", apiPose.position.x},
                            {"y", apiPose.position.y},
                            {"z", apiPose.position.z}
                        }},
                        {"orientation", {
                            {"w", apiPose.orientation.w},
                            {"x", apiPose.orientation.x},
                            {"y", apiPose.orientation.y},
                            {"z", apiPose.orientation.z}
                        }}
                    };
                    jsonTrail.push_back(jsonPose);
                }
            }
            nextOutput->additionalData = additionalOutputInfo.dump();
        }

        if (onOutput) onOutput(nextOutput);
        if (recorder) {
            Pose pose = nextOutput->pose;
            // JSONL format orientation is device-to-world.
            pose.orientation.x = -pose.orientation.x;
            pose.orientation.y = -pose.orientation.y;
            pose.orientation.z = -pose.orientation.z;
            Vector3d velocity = nextOutput->velocity;
            recorder->addOdometryOutput(pose, velocity);
        }
        outputLock.unlock();
    }

    /**
     * Automatically create matching accelerated-arrays image and operation
     * factories corresponding to the type of given input images
     */
    void ensureFactories(Image::StorageType imageType) {
        assert(imageProcessingQueue);
        if (accImageFactory) return;

        if (imageType == Image::StorageType::CPU) {
            log_debug("Initializing CPU image processing");
            accImageFactory = accelerated::cpu::Image::createFactory();
            // NOTE: this could also use a thread pool for the CPU processing
            accOpsFactory =  accelerated::cpu::operations::createFactory(*imageProcessingQueue);
        } else {
        #ifdef DAZZLING_GPU_ENABLED
            log_debug("Initializing GPU image processing");
            accImageFactory = accelerated::opengl::Image::createFactory(*imageProcessingQueue);
            accOpsFactory =  accelerated::opengl::operations::createFactory(*imageProcessingQueue);
        #else
            assert(false && "not built with GPU support");
        #endif
        }
        trackerImageFactory = tracker::CpuImage::buildFactory(*imageProcessingQueue, *accImageFactory, *accOpsFactory, parameters.api.parameters);
    }

    std::shared_ptr<Image> copyAsColorFrame(Image &input) {
        ensureFactories(input.storageType);
        if (!colorFrameBuffer) {
            // lazy initialize color frame buffers
            const int w = input.width, h = input.height;
            rgbaSpec.reset(new accelerated::ImageTypeSpec(accImageFactory->getSpec(4, accelerated::ImageTypeSpec::DataType::UFIXED8)));
            colorFrameBuffer = std::make_unique< util::Allocator<accelerated::Image> >([this, w, h]() {
                return accImageFactory->create(w, h, rgbaSpec->channels, rgbaSpec->dataType);
            });

            std::string swiz;
            switch (input.channels) {
                case 1:
                    swiz = "rrr1";
                    break;
                case 3:
                    swiz = "rgb1";
                    break;
                case 4:
                    swiz = "rgba";
                    break;
                default:
                    assert(false);
                    break;
            }
            log_debug("input -> color frame swizzle: %s", swiz.c_str());
            copyAsColorFrameOp = accOpsFactory->swizzle(swiz).build(input, *rgbaSpec);
        }

        auto colorBuf = colorFrameBuffer->next();
        // log_debug("converting input -> color");
        accelerated::operations::callUnary(copyAsColorFrameOp, input, *colorBuf);

        // this operation should be called only from the OpenGL thread and
        // this processes all pending GL operations, including the copy /
        // color convert ops above (in case of GPU images)
        imageProcessingQueue->processAll();

        return colorBuf;
    }

    std::unique_ptr<tracker::Image> copyAsGrayFrame(Image &input, std::shared_ptr<const tracker::Camera> camera) {
        // log_debug("converting input -> gray");
        ensureFactories(input.storageType);
        auto img = trackerImageFactory->build(input, camera);
        imageProcessingQueue->processAll();
        return img;
    }

    cv::Mat getAsCvMat(Image &image) {
        cv::Mat mat;
        accelerated::opencv::copy(image, mat);
        imageProcessingQueue->processAll();
        return mat;
    }

    void visualizeCovariance(cv::Mat &target, bool correlation) {
        assert(control);
        Eigen::MatrixXd P;
        int mapPointOffset;
        controlProcessingQueue.enqueueAndWait([this, &mapPointOffset, &P]() {
            assert(control);
            P = control->getEKF().getStateCovariance();
            mapPointOffset = control->getEKF().getMapPointStateIndex(0);
        });
        odometry::visualizeCovariance(target, P, mapPointOffset, correlation);
    }

    api_visualization_helpers::VisualizationHelper* getVisualizationHelper() final {
        return visuData.get();
    }

    nlohmann::json additionalOutputInfo = R"({
        "positionCovariance": null,
        "focalLength": 0.0,
        "biasMean": {
            "gyroscopeAdditive": null,
            "accelerometerAdditive": null,
            "accelerometerTransform": null},
        "biasCovarianceDiagonal": {
            "gyroscopeAdditive": null,
            "accelerometerAdditive": null,
            "accelerometerTransform": null},
        "stationary": false,
        "stationaryVisual": false,
        "poseTrail": []
    })"_json;

    bool enableVisualization = false;
    // Not const due to setParameterString.
    // Avoid modifying on the fly or suffer the consequences
    DebugParameters parameters;

    PoseHistoryMap poseHistories;
    util::GpsToLocalConverter gpsToLocal;
    // OutputBuffer outputBuffer;
    // The visualizations / text views can call things like api.getPose()
    // from unwiedly places. Easiest to handle if the output is protected
    // by its own mutex
    std::mutex outputMutex;

    // visualization stuff. should be only accessed from the GL thread
    // in case of GPU processing
    std::unique_ptr<NotifyingQueue> imageProcessingQueue;
    std::unique_ptr<accelerated::Image::Factory> accImageFactory;
    std::unique_ptr<accelerated::operations::StandardFactory> accOpsFactory;
    std::unique_ptr<accelerated::ImageTypeSpec> rgbaSpec;
    accelerated::operations::Function copyAsColorFrameOp;
    std::unique_ptr<tracker::CpuImage::Factory> trackerImageFactory;
    std::unique_ptr< util::Allocator<accelerated::Image> > colorFrameBuffer;
    std::unique_ptr<api_visualization_helpers::VisualizationHelper> visuData;

    // should be destroyed before the buffers
    std::unique_ptr<util::Allocator< Output >> outputStore;

    std::unique_ptr<odometry::Control> control;

    std::unique_ptr<recorder::Recorder> recorder;

    util::BoundedProcessingQueue controlProcessingQueue;

};

std::string InternalAPI::Output::asJson() const {
    return outputToJson(*this, false);
}

std::unique_ptr<api::InternalAPI> buildVio(const InternalAPI::DebugParameters &parameters) {
    return std::unique_ptr<InternalAPI>(new InternalAPIImplementation(parameters));
}

std::unique_ptr<api::VioApi> buildVio(std::istream &calibrationJson, std::istream &configYaml) {
    CommandLineParameters cmd(0, nullptr);
    // TODO: move parse_calibration_json elsewhere and stop using CommandLineParameters here.
    cmd.parse_calibration_json(calibrationJson);
    cmd.parse_yaml_config(configYaml);
    api::InternalAPI::DebugParameters debugParameters;
    debugParameters.api.parameters = cmd.parameters;
    return std::unique_ptr<InternalAPI>(new InternalAPIImplementation(debugParameters));
}

}
