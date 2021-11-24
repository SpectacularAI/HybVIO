#ifndef VISUALIZATION_INTERNALS_HPP
#define VISUALIZATION_INTERNALS_HPP

#include <Eigen/Dense>

// forward declarations
namespace cv { class Mat; }

namespace odometry {

struct TrackVisualization;

// internal visualization helpers
void visualizeTrack(cv::Mat& colorFrame, const TrackVisualization& track, bool secondCamera);
void visualizeCovariance(cv::Mat& corrFrame, const Eigen::MatrixXd& P, int hybridMapSize, bool correlation);
}

#endif
