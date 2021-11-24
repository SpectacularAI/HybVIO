#ifndef TRACKER_FEATURE_DETECTOR_H_
#define TRACKER_FEATURE_DETECTOR_H_

#include <memory>
#include <vector>
#include <accelerated-arrays/image.hpp>
#include "track.hpp"

namespace cv { class Mat; }
namespace accelerated {
    struct Future;
    struct Processor;
    namespace operations { struct StandardFactory; }
}
namespace odometry { struct ParametersTracker; }

namespace tracker {
struct Image;

class FeatureDetector {
public:
    static std::unique_ptr<FeatureDetector> build(
        int imageWidth, int imageHeight,
        accelerated::Processor &proc,
        accelerated::Image::Factory &ifac,
        accelerated::operations::StandardFactory &ofac,
        const odometry::ParametersTracker &parameters);

    static std::unique_ptr<FeatureDetector> buildLegacyFAST(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &parameters);

    static std::unique_ptr<FeatureDetector> buildLegacyGFTT(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &parameters);

    virtual ~FeatureDetector();

    // sync version for CPU
    virtual void detect(
        Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) = 0;

    // async version for GPU
    virtual accelerated::Future detect(
        accelerated::Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) = 0;

    virtual bool supportsAsync() const {
        return false;
    }

    /**
     * Sync operation on the CPU: apply new mask to existing points.
     * These two sequences should be more-or-less the same
     *
     *      detect(image, corners, prevCorners, r).wait();
     *
     * OR
     *
     *      detect(image, corners, {}, 0).wait();
     *      applyMinDistance(corners, prevCorners, r);
     */
    void applyMinDistance(
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int minDistance) const;

    virtual void debugVisualize(cv::Mat &target) = 0;

protected:
    FeatureDetector(const odometry::ParametersTracker &parameters);
    const odometry::ParametersTracker &parameters;
};

}

#endif
