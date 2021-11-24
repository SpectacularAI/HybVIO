#ifndef TRACKER_STEREO_DISPARITY_H_
#define TRACKER_STEREO_DISPARITY_H_

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace odometry { struct ParametersTracker; }
namespace accelerated { struct Image; }

namespace tracker {
class StereoDisparity {
public:
    using Image = accelerated::Image;
    using PointCloud = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
    static std::unique_ptr<StereoDisparity> build(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &parameters);

    virtual ~StereoDisparity();
    virtual std::unique_ptr<Image> initializeDisparityVisualization() const = 0;
    virtual std::unique_ptr<Image> buildDisparityImage() const = 0;

    virtual void computeDisparity(Image &rectifiedFirst, Image &rectifiedSecond, Image &out) = 0;
    virtual float getDepth(const Eigen::Matrix4d &Q, accelerated::Image &disparity, const Eigen::Vector2f &pixCoords) const = 0;
    virtual void computePointCloud(const Eigen::Matrix4d &Q, Image &disparity, PointCloud &out) = 0;

    virtual void visualizeDisparity(Image &disparity, Image &out) const = 0;
    virtual void visualizeDisparityDepth(const Eigen::Matrix4d &Q, Image &disparity, Image &out) const = 0;
};
}

#endif
