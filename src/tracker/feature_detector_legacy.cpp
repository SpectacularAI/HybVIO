#include "feature_detector.hpp"
#include "track.hpp"
#include "image.hpp"

#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"
#include <opencv2/opencv.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

namespace tracker {
namespace {
std::vector<cv::Point2f> &asOpenCv(std::vector<Feature::Point> &pv) {
    return reinterpret_cast<std::vector<cv::Point2f>&>(pv);
}

const std::vector<cv::Point2f> &asOpenCv(const std::vector<Feature::Point> &pv) {
    return reinterpret_cast<const std::vector<cv::Point2f>&>(pv);
}

cv::Range cappedRangeAround(int middle, int radius, int min, int max) {
    assert(middle >= min);
    assert(middle <= max);
    int left = min;
    int right = max;
    if (middle - radius > min) {
        left = middle - radius;
    }
    if (middle + radius < max) {
        right = middle + radius;
    }
    return cv::Range(left, right);
}

void drawMask(
    cv::Mat &mask,
    const std::vector<cv::Point2f>& corners,
    int maskRadius)
{
    assert(maskRadius >= 1);
    mask.setTo(cv::Scalar(1));
    for (size_t i = 0; i < corners.size(); i++) {
        // Place a square mask on each feature point.
        float x = corners[i].x;
        float y = corners[i].y;
        if (x > 0.0f && x < mask.cols && y > 0.0f && y < mask.rows) {
            cv::Range rx = cappedRangeAround(static_cast<int>(x), maskRadius, 0, mask.cols - 1);
            cv::Range ry = cappedRangeAround(static_cast<int>(y), maskRadius, 0, mask.rows - 1);
            mask(ry, rx) = 0;
        }
    }
}

class MaskedFeatureDetector : public FeatureDetector {
protected:
    int maskRadius = 0;
    cv::Mat mask;
    cv::Ptr<cv::FeatureDetector> detector;

    // workspace
    std::vector<cv::KeyPoint> keypoints;

    virtual void detectNonMasked(const cv::Mat& image, std::vector<cv::Point2f>& corners)
    {
        keypoints.clear();
        corners.clear();
        detector->detect(image, keypoints, mask);
        cv::KeyPoint::convert(keypoints, corners);
    }

    void detect(
        const cv::Mat &cvImg,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius)
    {
        setMask(prevCorners, maskRadius);
        detectNonMasked(cvImg, asOpenCv(corners));
    }

    void setMask(const std::vector<Feature::Point>& corners, int radius) {
        maskRadius = radius;
        drawMask(mask, asOpenCv(corners), maskRadius);
    }

public:
    MaskedFeatureDetector(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &params)
    : FeatureDetector(params) {
        mask = cv::Mat(cv::Size(imageWidth, imageHeight), CV_8U, cv::Scalar(1));
    }

    void detect(
        Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) final
    {
        detect(reinterpret_cast<CpuImage&>(image).getOpenCvMat(), corners, prevCorners, maskRadius);
    }

    accelerated::Future detect(
        accelerated::Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) final
    {
        detect(accelerated::opencv::ref(image), corners, prevCorners, maskRadius);
        return accelerated::Future::instantlyResolved();
    }

    void debugVisualize(cv::Mat &target) final {
        (void)target;
        assert(false && "not supported");
    }
};

class GFTTFeatureDetector : public MaskedFeatureDetector {
public:
    GFTTFeatureDetector(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &params)
    : MaskedFeatureDetector(imageWidth, imageHeight, params) {

        const auto minDim = std::min(imageWidth, imageHeight);
        const double su = minDim / 720.0;

        auto d = cv::GFTTDetector::create();
        d->setMaxFeatures(parameters.maxTracks);
        d->setQualityLevel(parameters.gfttQualityLevel);
        // Simple scaling factor which works well for 16:9 or more square aspect ratios.
        d->setMinDistance(parameters.gfttMinDistance * su);
        d->setBlockSize(parameters.gfttBlockSize);
        d->setHarrisDetector(false);
        d->setK(parameters.gfttK);

        detector = d;
    }
};

class CustomFastFeatureDetector : public MaskedFeatureDetector {
public:
    CustomFastFeatureDetector(
        int imageWidth, int imageHeight,
        const odometry::ParametersTracker &params)
    : MaskedFeatureDetector(imageWidth, imageHeight, params) {

        auto d = cv::FastFeatureDetector::create();
        d->setNonmaxSuppression(true);

        detector = d;
    }

protected:
    void detectNonMasked(const cv::Mat& image, std::vector<cv::Point2f>& corners) final {
        keypoints.clear();
        corners.clear();
        detector->detect(image, keypoints, mask);

        // use strongest keypoints first
        std::stable_sort(keypoints.begin(), keypoints.end(),
            [](const cv::KeyPoint &a, const cv::KeyPoint &b) -> bool {
                return a.response > b.response;
            });

        corners.clear();
        corners.reserve(keypoints.size());
        for (const auto &kp : keypoints) corners.push_back(kp.pt);
        applyMinDistance(
            reinterpret_cast<std::vector<Feature::Point>&>(corners),
            {},
            maskRadius);
    }
};
}

void FeatureDetector::applyMinDistance(
    std::vector<Feature::Point>& corners,
    const std::vector<Feature::Point>& prevCorners,
    int r) const
{
    // in-place filter
    std::size_t nOut = 0;
    const float r2 = static_cast<float>(r * r);
    for (const auto &c : corners) {
        bool nearOther = false;
        if (r > 0) {
            // TODO: could DRY a bit
            for (const auto &other : prevCorners) {
                const float dx = other.x - c.x, dy = other.y - c.y;
                if (dx*dx + dy*dy < r2) {
                    nearOther = true;
                    break;
                }
            }
            if (!nearOther) {
                for (std::size_t i = 0; i < nOut; ++i) {
                    const auto &other = corners.at(i);
                    const float dx = other.x - c.x, dy = other.y - c.y;
                    if (dx*dx + dy*dy < r2) {
                        nearOther = true;
                        break;
                    }
                }
            }
        }
        if (!nearOther) {
            corners.at(nOut++) = c;
        }
        if (int(nOut) >= parameters.maxTracks) break;
    }
    corners.resize(nOut);
}

std::unique_ptr<FeatureDetector> FeatureDetector::buildLegacyFAST(
        int w, int h, const odometry::ParametersTracker &p)
{
    return std::unique_ptr<FeatureDetector>(new CustomFastFeatureDetector(w, h, p));
}

std::unique_ptr<FeatureDetector> FeatureDetector::buildLegacyGFTT(
        int w, int h, const odometry::ParametersTracker &p)
{
    return std::unique_ptr<FeatureDetector>(new GFTTFeatureDetector(w, h, p));
}

}
