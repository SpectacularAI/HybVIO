#ifndef TRACKER_IMAGE_H_
#define TRACKER_IMAGE_H_

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cstdint>
#include <utility>

// TODO: clarify headers in accelerated-arrays
#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/standard_ops.hpp>

#include "track.hpp"

namespace cv { class Mat; }
namespace odometry { struct Parameters; }
namespace accelerated { struct Queue; struct Processor; }

namespace tracker {
class Camera;

/**
 * An interface to image data which is meant to efficiently support both
 * CPU and GPU implementations
 */
struct Image {
    typedef std::vector<Feature::Point> KeypointList;
    typedef std::vector<Feature::Status> KeypointStatusList;
    typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> StereoPointCloud;

    virtual void findKeypoints(
        const KeypointList &existingMask, // may or may not be neeeded
        int maskRadius,
        KeypointList &out) = 0;

    // LK tracking
    virtual void opticalFlow(
        Image &prev,
        const KeypointList &keypointsIn,
        KeypointList &keypointsOut,
        KeypointStatusList &status,
        bool useInitialCorners,
        int overrideMaxIterations = -1) = 0;

    virtual void computeDisparity(Image &secondImage) = 0;
    virtual bool hasStereoPointCloud() const = 0;
    virtual const StereoPointCloud &getStereoPointCloud() = 0;
    // returns -1 if depth is not available at the given pixel coordinates for any reason
    virtual float getDepth(const Eigen::Vector2f &pixCoords) const = 0;

    virtual ~Image() = default;
    Image(int w, int h) : width(w), height(h) {};
    const int width, height;

    virtual accelerated::Image &getAccImage() = 0;
    virtual std::shared_ptr<const Camera> getCamera() const = 0;

    // note: not the most elegant solution using "image" instances to deliver
    // these factories to other classes, but quite handy
    virtual accelerated::Processor &getProcessor() = 0;
    virtual accelerated::Image::Factory &getImageFactory() = 0;
    virtual accelerated::operations::StandardFactory &getOperationsFactory() = 0;

    // debug visualization stuff
    enum class VisualizationMode { CORNER_MEASURE, DISPARITY, DEPTH };
    virtual void debugVisualize(cv::Mat &target, VisualizationMode mode) const = 0;

    struct Factory {
        using ImagePtr = std::unique_ptr<Image>;
        virtual ImagePtr build(accelerated::Image &image, std::shared_ptr<const Camera> camera) = 0;

        virtual std::pair<ImagePtr, ImagePtr> buildStereo(
            accelerated::Image &firstImage,
            accelerated::Image &secondImage,
            std::shared_ptr<const Camera> firstCamera,
            std::shared_ptr<const Camera> secondCamera) = 0;

        virtual ~Factory() = default;
    };

    static std::unique_ptr<Factory> buildFactory(
        accelerated::Queue &imageProcessingQueue,
        accelerated::Image::Factory &ifac,
        accelerated::operations::StandardFactory &ofac,
        const odometry::Parameters &params);
};

struct CpuImage : Image {
    CpuImage(int w, int h) : Image(w, h) {}
    virtual const cv::Mat &getOpenCvMat() = 0;
};

// TODO GpuImage

}

#endif
