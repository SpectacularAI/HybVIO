#ifndef TRACKER_UTIL_H_
#define TRACKER_UTIL_H_

#include "track.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"

#include <ctime>
#include <string>

namespace cv { class Mat; }

namespace tracker {
namespace util {

void automaticCameraParametersWhereUnset(odometry::Parameters &parameters);
void scaleImageParameters(odometry::ParametersTracker &parameters, double scaleFactor);
void printTimeSince(const std::clock_t &since, const std::string &description = "took");
double timeSince(const std::clock_t &since);
void rotateMatrixCW90(const cv::Mat &source, cv::Mat &dest, int times);
void matchIntensities(cv::Mat &l, cv::Mat &r, double weight = 1.0);

} // namespace util
} // namespace tracker

#endif // TRACKER_UTIL_H_
