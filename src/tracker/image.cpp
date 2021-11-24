#include "camera.hpp"
#include "image.hpp"
#include "image_pyramid.hpp"
#include "optical_flow.hpp"
#include "feature_detector.hpp"
#include "undistorter.hpp"
#include "stereo_disparity.hpp"
#include "stereo_rectifier.hpp"
#include "subpixel_adjuster.hpp"

#include <cassert>
#include <opencv2/core.hpp>
#include "../util/allocator.hpp"
#include "../odometry/parameters.hpp"

#include <accelerated-arrays/function.hpp>
#include <accelerated-arrays/future.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

namespace tracker {
namespace {
class ImageImplementation : public CpuImage {
public:
    struct SharedData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        const odometry::ParametersTracker &parameters;
        accelerated::Queue &imageProcessingQueue;
        accelerated::Image::Factory &imageFactory;
        accelerated::operations::StandardFactory &opFactory;

        const std::unique_ptr<FeatureDetector> detector;
        const std::unique_ptr<SubPixelAdjuster> subPix;
        const std::unique_ptr<ImagePyramid::Factory> pyramidFactory;
        const std::unique_ptr<OpticalFlow> opticalFlow;
        const std::unique_ptr<StereoDisparity> stereoDisparity;
        util::Allocator<std::vector<Feature::Point>> cornerBuffer;
        util::Allocator<StereoPointCloud> stereoPointCloudBuffer;
        util::Allocator<accelerated::Image> disparityImageFactory;

        Eigen::Matrix4d disparityToDepthQ;

        SharedData(int width, int height,
            const odometry::ParametersTracker &p,
            accelerated::Queue &queue,
            accelerated::Image::Factory &imageFactory,
            accelerated::operations::StandardFactory &opFactory)
        :
            parameters(p),
            imageProcessingQueue(queue),
            imageFactory(imageFactory),
            opFactory(opFactory),
            detector(FeatureDetector::build(width, height, imageProcessingQueue, imageFactory, opFactory, p)),
            subPix(p.subPixMaxIter > 0 ? SubPixelAdjuster::build(p) : nullptr),
            pyramidFactory(ImagePyramid::Factory::buildOpenCv(p)),
            opticalFlow(OpticalFlow::buildOpenCv(p)),
            stereoDisparity(StereoDisparity::build(width, height, p)),
            cornerBuffer([this](){
                auto vec = std::make_unique< std::vector<Feature::Point> >();
                vec->resize(parameters.maxTracks);
                return vec;
            }),
            stereoPointCloudBuffer([](){ return std::make_unique<StereoPointCloud>(); }),
            disparityImageFactory([this](){ return stereoDisparity->buildDisparityImage(); }),
            disparityToDepthQ(Eigen::Matrix4d::Zero())
        {}
    };

    void findKeypoints(
        const KeypointList &existingMask,
        int maskRadius,
        KeypointList &out) final
    {
        if (pendingFeatureDetection) {
            pendingFeatureDetection->wait();
            out = *pendingCorners;
            shared.detector->applyMinDistance(out, existingMask, maskRadius);
        } else {
            shared.detector->detect(*this, out, existingMask, maskRadius);
        }
        if (shared.subPix) {
            ensureCpuImageAsync().wait();
            shared.subPix->adjust(*this, out);
        }
    }

    void opticalFlow(
        Image &prev,
        const KeypointList &keypointsIn,
        KeypointList &keypointsOut,
        KeypointStatusList &status,
        bool useInitialCorners,
        int overrideMaxIterations) final
    {
        ensureImagePyramid();
        auto &prevCpu = reinterpret_cast<ImageImplementation&>(prev);
        prevCpu.ensureImagePyramid();
        shared.opticalFlow->compute(
            *prevCpu.imagePyramid,
            *imagePyramid,
            keypointsIn,
            keypointsOut,
            status,
            useInitialCorners,
            overrideMaxIterations);
    }

    void computeDisparity(Image &secondImage) final {
        // This is not necessary if the images are pre-rectified. This can be commented out in that case
        assert(shared.parameters.useRectification && "must rectify before computing stereo disparity");
        if (disparity) return;

        disparity = shared.disparityImageFactory.next();

        // Currently only supported on the CPU, even if rectification was done on the GPU
        shared.stereoDisparity->computeDisparity(
            *accelerated::opencv::ref(getOpenCvMat()),
            *accelerated::opencv::ref(reinterpret_cast<CpuImage&>(secondImage).getOpenCvMat()),
            *disparity);
    }

    bool hasStereoPointCloud() const final {
        return !!stereoPointCloud;
    }

    const StereoPointCloud &getStereoPointCloud() final {
        if (!stereoPointCloud) {
            assert(disparity);
            // note: would not safe to call anywhere on the GPU
            stereoPointCloud = shared.stereoPointCloudBuffer.next();
            shared.stereoDisparity->computePointCloud(shared.disparityToDepthQ, *disparity, *stereoPointCloud);
        }
        return *stereoPointCloud;
    }

    float getDepth(const Eigen::Vector2f &pixCoords) const final {
        if (!disparity) return -1;
        return shared.stereoDisparity->getDepth(shared.disparityToDepthQ, *disparity, pixCoords);
    }

    const cv::Mat &getOpenCvMat() final {
        ensureCpuImageAsync().wait();
        return image;
    }

    void debugVisualize(cv::Mat &target, VisualizationMode mode) const final {
        switch (mode) {
        case VisualizationMode::CORNER_MEASURE:
            shared.detector->debugVisualize(target);
            break;
        case VisualizationMode::DISPARITY:
            {
                if (!disparity) return;
                auto targetAcc = accelerated::opencv::ref(target);
                shared.stereoDisparity->visualizeDisparity(*disparity, *targetAcc);
            }
            break;
        case VisualizationMode::DEPTH:
            {
                if (!disparity) return;
                auto targetAcc = accelerated::opencv::ref(target);
                shared.stereoDisparity->visualizeDisparityDepth(shared.disparityToDepthQ, *disparity, *targetAcc);
            }
            break;
        }
    }

    ImageImplementation(SharedData &data, std::shared_ptr<accelerated::Image> img, std::shared_ptr<const Camera> camera)
    : CpuImage(img->width, img->height), shared(data), accImage(img), camera(camera), cpuCopyCompleted(false)
    {
        if (img->storageType == accelerated::Image::StorageType::CPU) {
            // If this is a CPU image, we can use the OpenCV image as a reference
            // without a copy
            image = accelerated::opencv::ref(*accImage);
        } else {
            ensureCpuImageAsync();
        }

        if (shared.detector->supportsAsync()) {
            pendingCorners = shared.cornerBuffer.next();
            // Launch async unmasked feature detection right away (on the GPU)
            auto fut = shared.detector->detect(*accImage, *pendingCorners, {}, 0);
            pendingFeatureDetection.reset(new accelerated::Future(fut));
        }
    }

    std::shared_ptr<const Camera> getCamera() const final {
        return camera;
    }

    accelerated::Image &getAccImage() final {
        assert(accImage);
        return *accImage;
    }

    accelerated::Processor &getProcessor() {
        return shared.imageProcessingQueue;
    }

    accelerated::Image::Factory &getImageFactory() {
        return shared.imageFactory;
    }

    accelerated::operations::StandardFactory &getOperationsFactory() {
        return shared.opFactory;
    }

private:
    void ensureImagePyramid() {
        if (!imagePyramid) {
            std::shared_ptr<accelerated::Image> cpuImg = accelerated::opencv::ref(getOpenCvMat());
            imagePyramid = shared.pyramidFactory->compute(cpuImg);
        }
    }

    accelerated::Future ensureCpuImageAsync() {
        if (accImage->storageType == accelerated::Image::StorageType::CPU || cpuCopyCompleted.load()) {
            assert(!image.empty());
            return accelerated::Future::instantlyResolved();
        }

        // launch copy only once
        if (!cpuCopyLaunched.test_and_set()) accelerated::opencv::copy(*accImage, image);
        return shared.imageProcessingQueue.enqueue([this] { cpuCopyCompleted.store(true); });
    }

    SharedData &shared;
    cv::Mat image;
    std::shared_ptr<ImagePyramid> imagePyramid;
    std::shared_ptr<accelerated::Image> accImage;
    std::shared_ptr<const Camera> camera;

    std::atomic_flag cpuCopyLaunched = ATOMIC_FLAG_INIT;
    std::atomic<bool> cpuCopyCompleted;
    std::unique_ptr<accelerated::Future> pendingFeatureDetection;
    std::shared_ptr<std::vector<Feature::Point>> pendingCorners;
    std::shared_ptr<StereoPointCloud> stereoPointCloud;
    std::shared_ptr<accelerated::Image> disparity;
};

class FactoryImplementation : public Image::Factory {
public:
    ImagePtr build(accelerated::Image &inputImage, std::shared_ptr<const Camera> camera) final {
        if (!data) initialize(inputImage, camera, {});
        return buildPrivate(inputImage, camera, undistorters[0].get());
    }

    std::pair<ImagePtr, ImagePtr> buildStereo(
        accelerated::Image &firstImage,
        accelerated::Image &secondImage,
        std::shared_ptr<const Camera> firstCamera,
        std::shared_ptr<const Camera> secondCamera) final
    {
        assert(secondImage.width == firstImage.width && secondImage.height == firstImage.height);
        if (!data) initialize(firstImage, firstCamera, secondCamera);
        return std::make_pair(
            buildPrivate(firstImage, firstCamera, undistorters[0].get()),
            buildPrivate(secondImage, secondCamera, undistorters[1].get()));
    }

    FactoryImplementation(
        accelerated::Queue &queue,
        accelerated::Image::Factory &imageFactory,
        accelerated::operations::StandardFactory &opFactory,
        const odometry::Parameters &params)
    :
        parameters(params),
        imageProcessingQueue(queue),
        imageFactory(imageFactory),
        opFactory(opFactory)
    {}

private:
    ImagePtr buildPrivate(accelerated::Image &inputImage, std::shared_ptr<const Camera> camera, Undistorter *undistorter) {
        assert(data);

        accelerated::Image *undistorerInput = nullptr;
        if (colorToGrayOp) {
            // use member field to keep object alive until the async GPU
            // operations (rectification) have completed
            tmpImage = frameBuffer->next();
            accelerated::operations::callUnary(colorToGrayOp, inputImage, *tmpImage);
            undistorerInput = tmpImage.get();
        } else {
            assert(undistorter);
            undistorerInput = &inputImage;
        }

        std::shared_ptr<const Camera> resultCamera;
        std::shared_ptr<accelerated::Image> resultImage;
        if (undistorter) {
            auto rectified = undistorter->undistort(*undistorerInput, camera);
            resultCamera = rectified.camera;
            resultImage = rectified.image;
        } else {
            assert(tmpImage);
            resultImage = tmpImage;
            resultCamera = camera;
            tmpImage = {};
        }

        // On CPU, complete the color to gray op before proceeding in case the
        // ImageImplementation would like to do something syncrhonously that
        // assumes the data to be ready
        if (inputImage.storageType == accelerated::Image::StorageType::CPU) imageProcessingQueue.processAll();

        return std::unique_ptr<CpuImage>(new ImageImplementation(*data, resultImage, resultCamera));
    }

    void initialize(
        const accelerated::Image &inputImage,
        std::shared_ptr<const Camera> camera,
        std::shared_ptr<const Camera> secondCamera)
    {
        const int w = inputImage.width, h = inputImage.height;
        data.reset(new ImageImplementation::SharedData(w, h, parameters.tracker,
            imageProcessingQueue, imageFactory, opFactory));

        if (secondCamera) {
            std::array<api::CameraParameters, 2> intrinsics;
            intrinsics[0] = camera->getIntrinsic();
            intrinsics[1] = secondCamera->getIntrinsic();
            rectifier = StereoRectifier::build(w, h, intrinsics, parameters);

            if (rectifier) {
                log_debug("stereo rectifier enabled");
                auto rectifiedCameras = rectifier->getRectifiedCameras();
                for (size_t i=0; i<2; ++i) {
                    undistorters[i] = Undistorter::buildRectified(w, h, rectifiedCameras[i], imageFactory, opFactory, parameters.tracker);
                }
                data->disparityToDepthQ = rectifier->getDepthQMatrix();
            }
        } else {
            undistorters[0] = Undistorter::buildMono(
                w, h, camera->getFocalLength(), imageFactory, opFactory, parameters.tracker);
            if (undistorters[0]) log_debug("mono undistortion enabled");
        }

        const auto graySpec = imageFactory.getSpec(1, accelerated::ImageTypeSpec::DataType::UFIXED8);

        frameBuffer = std::make_unique< util::Allocator<accelerated::Image> >([this, graySpec, w, h]() {
            return imageFactory.create(w, h, graySpec.channels, graySpec.dataType);
        });

        if (inputImage.channels == 1) {
            if (undistorters[0]) {
                log_debug("direct undistorter input");
            } else {
                log_debug("input is gray: direct copy");
                colorToGrayOp = opFactory.copy().build(inputImage, graySpec);
            }
        }
        else {
            log_debug("input is BGR/RGB(A), %d x %d. Converting color -> gray", w, h);
            // TODO: could parametrize this
            // cf. https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            // NOTE: if the input is an OpenCV image in BGR(A) format, this
            // vector should actually be reversed. However, the difference
            // in performance should be negligible
            std::vector< double > coeff = { 0.299, 0.587, 0.114 };
            if (inputImage.channels == 4) {
                coeff.push_back(0);
            } else {
                assert(inputImage.channels == 3);
            }
            colorToGrayOp = opFactory.pixelwiseAffine({ coeff }).build(inputImage, graySpec);
        }
    }

    std::unique_ptr<ImageImplementation::SharedData> data;

    const odometry::Parameters &parameters;
    accelerated::Queue &imageProcessingQueue;
    accelerated::Image::Factory &imageFactory;
    accelerated::operations::StandardFactory &opFactory;
    std::unique_ptr< util::Allocator<accelerated::Image> > frameBuffer;
    std::shared_ptr<accelerated::Image> tmpImage;
    accelerated::operations::Function colorToGrayOp;
    std::array<std::unique_ptr<Undistorter>, 2> undistorters;
    std::unique_ptr<StereoRectifier> rectifier;
};
}

std::unique_ptr<Image::Factory> Image::buildFactory(
    accelerated::Queue &q,
    accelerated::Image::Factory &ifac,
    accelerated::operations::StandardFactory &ofac,
    const odometry::Parameters &params)
{
    return std::unique_ptr<Image::Factory>(new FactoryImplementation(q, ifac, ofac, params));
}
}
