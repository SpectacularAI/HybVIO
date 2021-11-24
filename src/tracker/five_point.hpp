#ifndef TRACKER_FIVE_POINT_H_
#define TRACKER_FIVE_POINT_H_

#include <opencv2/opencv.hpp>

namespace tracker {
/**
 * The matrices points1 and points2 are expected to have shape 2 * npoints
 * and data type CV_64F. Each point should be in normalized coordinates, i.e.,
 * (px, py) so that the camera ray would be parallel to (px, py, 1)
 */
cv::Mat findEssentialMatRansacMaxIter(
    const cv::Mat &points1, const cv::Mat &points2,
    double prob, double threshold,
    cv::OutputArray _mask, int maxIters);
} // namespace tracker

#endif // TRACKER_FIVE_POINT_H_
