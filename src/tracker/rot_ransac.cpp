#include "rot_ransac.hpp"

#include "track.hpp"
#include "camera.hpp"

const size_t ROT_RANSAC_MAX_ITERS = 100;

namespace tracker {
namespace rot_ransac {

struct RotRansac::impl {
    std::vector<cv::Matx31f> p1, p2;
    std::vector<size_t> inliers;
    std::vector<size_t> indsTmp;

    cv::Matx33f solveRotation(const std::vector<cv::Matx31f>& p1, const std::vector<cv::Matx31f>& p2, const std::array<size_t, 2>& inds) {
        indsTmp = {inds[0], inds[1]};
        return tracker::rot_ransac::solveRotation(p1, p2, indsTmp);
    }
};

static cv::Matx31f pixelToRay(Feature::Point pixel, const Camera &camera) {
    Eigen::Vector3d r;
    // Does not check success.
    camera.pixelToRay({ pixel.x, pixel.y }, r);
    return cv::Matx31f(r.x(), r.y(), r.z());
}

static bool rayToPixel(cv::Matx31f ray, const Camera &camera, Feature::Point &pixel) {
    Eigen::Vector2d p;
    if (!camera.rayToPixel(Eigen::Vector3d(ray(0), ray(1), ray(2)), p)) {
        return false;
    }
    pixel = { float(p.x()), float(p.y()) };
    return true;
}

// Fit a rotation model to two point sets using RANSAC to detect and discard outliers.
// If the focal length in the camera matrix `K` is too small (of the order 10^0),
// the algorithm might suffer from numerical instabilities.
cv::Matx33f RotRansac::fit(
    const std::vector<Feature::Point> &c1,
    const std::vector<Feature::Point> &c2,
    const Camera& camera1,
    const Camera& camera2,
    std::vector<Feature::Status> &bestInliers,
    std::mt19937& rng
) {
    assert(c1.size() == c2.size());

    // Transform image pixel coordinates into unit vectors of corresponding
    // projection direction defined by the camera matrix.
    size_t n = c1.size();

    auto &p1 = pimpl->p1;
    auto &p2 = pimpl->p2;
    p1.clear();
    p2.clear();
    p1.reserve(n);
    p2.reserve(n);
    for (size_t i = 0; i < n; i++) {
        p1.push_back(pixelToRay(c1[i], camera1));
        p2.push_back(pixelToRay(c2[i], camera2));
    }
    assert(p1.size() == n);
    assert(p2.size() == n);

    // Do RANSAC: Sample 2 pairs of vectors and fit the rotation model to them,
    // counting the number of inliers. After maximum number of iterations (or
    // a converge criterion, if implemented), pick the best fit.
    assert(n >= 2);
    std::array<size_t, 2> bestInds = {{ 0, 1 }};
    bestInlierCount = 0;
    for (size_t k = 0; k < ROT_RANSAC_MAX_ITERS; k++) {
        size_t ind1 = rng() % n;
        size_t ind2 = rng() % n;
        assert(ind1 < n && ind2 < n);
        if (ind1 == ind2) continue;
        std::array<size_t, 2> inds = {{ ind1, ind2 }};
        cv::Matx33f R = pimpl->solveRotation(p1, p2, inds);

        // Rotate all the unit vectors using the model and classify inliers.
        size_t inlierCount = 0;
        for (size_t i = 0; i < n; i++) {
            Feature::Point qp;
            if (rayToPixel(R * p1[i], camera2, qp) && withinInlierThreshold(c2[i], qp)) {
                inlierCount++;
            }
        }

        if (inlierCount > bestInlierCount) {
            bestInlierCount = inlierCount;
            bestInds = inds;
        }
        if (inlierCount == n) break;
    }

    cv::Matx33f bestR = pimpl->solveRotation(p1, p2, bestInds);

    // Refine the fit by recomputing the model from the inliers.
    // Seems to basically never change the inliers, but could be useful if
    // the rotation matrix is used for something else, for example comparing
    // to gyroscope measurements.
    pimpl->inliers.clear();
    for (size_t i = 0; i < n; i++) {
        Feature::Point qp;
        if (rayToPixel(bestR * p1[i], camera2, qp) && withinInlierThreshold(c2[i], qp)) {
            pimpl->inliers.push_back(i);
        }
    }
    if (pimpl->inliers.size() >= 2) {
        bestR = solveRotation(p1, p2, pimpl->inliers);
    }

    // Compute the final inliers in the desired format.
    for (size_t i = 0; i < n; i++) {
        Feature::Point qp;
        int ii = static_cast<int>(i);
        if (rayToPixel(bestR * p1[i], camera2, qp) && withinInlierThreshold(c2[i], qp)) {
            bestInliers.at(ii) = Feature::Status::TRACKED;
        }
        else {
            bestInliers.at(ii) = Feature::Status::RANSAC_OUTLIER;
        }
    }

    return bestR;
}

bool RotRansac::withinInlierThreshold(Feature::Point a, Feature::Point b) const {
    const double dx = a.x - b.x, dy = a.y - b.y;
    // Compare squares of the values.
    return (dx * dx + dy * dy) <= threshold_pow2;
}

// Fit a rotation to points about the origin. Returns a rotation matrix R that minimizes
// \sum_i ||p2_i - R*p1_i||^2. The input `inds` is used to mask the input.
// The implementation is adapted from the paper "Analysis of 3-D Rotation Fitting",
// by Kenichi Kanatani, 1994.
cv::Matx33f solveRotation(const std::vector<cv::Matx31f>& p1, const std::vector<cv::Matx31f>& p2, const std::vector<size_t>& inds) {
    assert(p1.size() == p2.size());
    assert(inds.size() <= p1.size());
    // The algorithm requires at least two pairs to fit a rotation.
    assert(inds.size() >= 2);

    cv::Mat It = cv::Mat::diag(cv::Mat({1, 1, -1}));
    It.convertTo(It, CV_32F);
    cv::Point3f m1, m2;

    cv::Matx33f H;
    for (size_t i = 0; i < inds.size(); i++) {
        size_t j = inds[i];
        assert(j < p1.size());
        H += p1[j] * p2[j].t();
    }

    cv::SVD svd(H, cv::SVD::MODIFY_A);

    cv::Mat R = svd.vt.t() * svd.u.t();
    if (cv::determinant(R) < 0) {
        R = svd.vt.t() * It * svd.u.t();
    }
    return R;
}

RotRansac::~RotRansac() = default;
RotRansac::RotRansac() :
    bestInlierCount(0),
    threshold_pow2(2.0 * 2.0), // Likely set later with information of the frame size.
    pimpl(new impl())
{}

} // namespace rot_ransac
} // namespace tracker
