#include "stereo_disparity.hpp"

#include "../odometry/parameters.hpp"

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <accelerated-arrays/cpu/image.hpp>
#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>
#include <cmath>

namespace tracker {
namespace {
constexpr float REL_MAX_DISPARITY = 0.1;
constexpr float MIN_DEPTH = 1.0;

constexpr float MIN_DISPARITY_FILTER = 1.0;
constexpr int SPECKLE_FILTER_RANGE = 10;
constexpr int SPECKLE_FILTER_WINDOW_SIZE = 500;

class OpenCvStereoDisparity : public StereoDisparity {
private:
    const odometry::ParametersTracker &parameters;
    const int width, height;
    std::unique_ptr<Image::Factory> imageFactory;
    int maxDisparity;
    cv::Ptr<cv::StereoMatcher> stereoMatcher;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    OpenCvStereoDisparity(
        int w, int h,
        const odometry::ParametersTracker &parameters)
    :
        parameters(parameters), width(w), height(h),
        imageFactory(accelerated::cpu::Image::createFactory())
    {
        maxDisparity = std::ceil(width * REL_MAX_DISPARITY / 32) * 32;
    }

    std::unique_ptr<Image> initializeDisparityVisualization() const final {
        // color visualization (RGBA/BRGA)
        return imageFactory->create<std::uint8_t, 4>(width, height);
    }

    std::unique_ptr<Image> buildDisparityImage() const final {
        return imageFactory->create<std::int16_t, 1>(width, height);
    }

    void computeDisparity(Image &rectifiedFirst, Image &rectifiedSecond, Image &out) final {
        if (!stereoMatcher) {
            // stereoMatcher = cv::StereoSGBM::create(0, maxDisparity);
            stereoMatcher = cv::StereoBM::create(maxDisparity);
            stereoMatcher->setSpeckleWindowSize(SPECKLE_FILTER_WINDOW_SIZE);
            stereoMatcher->setSpeckleRange(SPECKLE_FILTER_RANGE);
        }

        //auto stereoBM = cv::StereoBM::create(64);
        cv::Mat left = accelerated::opencv::ref(rectifiedFirst);
        cv::Mat right = accelerated::opencv::ref(rectifiedSecond);
        cv::Mat disparity = accelerated::opencv::ref(out);
        stereoMatcher->compute(left, right, disparity);
    }

    float getDepth(const Eigen::Matrix4d &Q, accelerated::Image &disparity, const Eigen::Vector2f &pixCoords) const {
        auto &dispCpu = accelerated::cpu::Image::castFrom(disparity);
        Eigen::Matrix4f Qfloat = Q.cast<float>();
        const int x = int(pixCoords.x()), y = int(pixCoords.y()); // could also interpolate
        if (x < 0 || y < 0 || x >= width || y >= height) return -1;

        auto v = dispCpu.get<std::int16_t>(x, y);
        float val = v / float(16);
        if (val <= 0 || val < MIN_DISPARITY_FILTER) return -1;
        const Eigen::Vector3f pos = (Qfloat * Eigen::Vector4f(x, y, val, 1)).hnormalized();
        return pos.norm(); // Note: this is really depth and not z coordinate
    }

    void computePointCloud(const Eigen::Matrix4d &Q, Image &disparity, PointCloud &out) {
        out.clear();
        auto &dispCpu = accelerated::cpu::Image::castFrom(disparity);
        int stride = parameters.stereoPointCloudStride;
        Eigen::Matrix4f Qfloat = Q.cast<float>();
        for (int y = 0; y < height; y += stride) {
            for (int x = 0; x < width; x += stride) {
                auto v = dispCpu.get<std::int16_t>(x, y);
                float val = v / float(16);
                if (val <= 0 || val < MIN_DISPARITY_FILTER) continue;
                const Eigen::Vector3f pos = (Qfloat * Eigen::Vector4f(x, y, val, 1)).hnormalized();
                if (pos.squaredNorm() < MIN_DEPTH*MIN_DEPTH) continue;
                out.push_back(pos);
            }
        }
    }

    void visualizeDisparity(Image &disp, Image &out) const {
        cv::Mat disparity = accelerated::opencv::ref(disp);
        cv::Mat visu = accelerated::opencv::ref(out);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                auto v = disparity.at<std::int16_t>(y, x);
                float val = v / float(16);
                visu.at<cv::Vec4b>(y, x) = colorMap(val, maxDisparity, MIN_DISPARITY_FILTER);
            }
        }
    }

    void visualizeDisparityDepth(const Eigen::Matrix4d &Q, Image &disp, Image &out) const {
        cv::Mat disparity = accelerated::opencv::ref(disp);
        cv::Mat visu = accelerated::opencv::ref(out);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                auto v = disparity.at<std::int16_t>(y, x);
                float val = v / float(16);

                Eigen::Vector4d ve(x, y, val, 1);
                Eigen::Vector4d de = Q * ve;
                float depth = de(2) / de(3);

                float MAX_DISP_DEPTH_M = 40;
                visu.at<cv::Vec4b>(y, x) = colorMap(depth, MAX_DISP_DEPTH_M, MIN_DISPARITY_FILTER);
            }
        }
    }

    cv::Vec4b colorMap(float val, float colorMax, float minFilter = 0) const {
        cv::Vec4b col;
        col(4) = 0xff;
        for (int c = 0; c < 3; ++c) {
            float max = colorMax / 3.0 * (c+1);
            float color = 0xff * val / max;
            col(2-c) = std::uint8_t(std::max(0, std::min(255, int(color))));
        }
        if (val <= 0) col = cv::Vec4b(0xaa, 0xaa, 0xaa, 0xff);
        else if (val < minFilter) col = cv::Vec4b(0, 0, 0xff, 0xff);
        return col;
    }
};
}

StereoDisparity::~StereoDisparity() = default;
std::unique_ptr<StereoDisparity> StereoDisparity::build(
    int w, int h,
    const odometry::ParametersTracker &parameters)
{
    return std::unique_ptr<StereoDisparity>(new OpenCvStereoDisparity(w, h, parameters));
}
}
