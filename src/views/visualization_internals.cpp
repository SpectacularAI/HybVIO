#include "visualization_internals.hpp"

#include "../odometry/util.hpp"
#include "../odometry/output.hpp"
#include "../odometry/ekf.hpp"
#include "../odometry/sample_sync.hpp"
#include "../odometry/tagged_frame.hpp"
#include "../util/logging.hpp"
#include "../api/vio.hpp"

#include <opencv2/opencv.hpp>
#include <limits>

namespace odometry {

//some generic colors
const cv::Scalar WHITE = cv::Scalar(255, 255, 255, 255);
const cv::Scalar DARK_RED = cv::Scalar(0, 0, 150, 255);
const cv::Scalar BLUE = cv::Scalar(255, 0, 0, 255);
const cv::Scalar MAGENTA = cv::Scalar(255, 0, 255, 255);
const cv::Scalar PURPLE = cv::Scalar(100, 0, 100, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255, 255);
const cv::Scalar BLACK = cv::Scalar(0, 0, 0, 255);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0, 255);

static void visualizeState(cv::Mat& frame, double scale, bool draw_state_lines, int hybridIndex);

static void visualizeTrack(cv::Mat& colorFrame, const Eigen::VectorXd& p, const cv::Scalar& color) {
    assert(p.size() % 2 == 0);
    if (p.size() == 0) return;

    double scale = static_cast<double>(std::max(colorFrame.cols, colorFrame.rows)) / 1280.0;
    int r = static_cast<int>(std::round(scale * 4.0));
    if (r <= 0) r = 1;
    int rLarge = static_cast<int>(std::round(scale * 10.0));
    if (rLarge <= 1) rLarge = 2;

    for (int i = 1; i < p.size() / 2; i++) {
        auto p1 = cv::Point2f(p(2 * (i - 1)), p(2 * (i - 1) + 1));
        auto p2 = cv::Point2f(p(2 * i), p(2 * i + 1));
        cv::line(colorFrame, p1, p2, color);
    }
    for (int i = 0; i < p.size() / 2; i++) {
        auto p1 = cv::Point2f(p(2 * i), p(2 * i + 1));
        cv::circle(colorFrame, p1, r, color, 1);
    }

    // Draw large circle in one end to help compare track directions.
    auto p1 = cv::Point2f(p(0), p(1));
    cv::circle(colorFrame, p1, rLarge, color, 1);
}

void visualizeTrack(cv::Mat &colorFrame, const TrackVisualization& track, bool secondCamera) {
    const Eigen::VectorXd& trackTracker = (secondCamera ? track.secondTrackTracker : track.trackTracker);
    const Eigen::VectorXd& trackProjection = (secondCamera ? track.secondTrackProjection : track.trackProjection);

    if (track.triangulateStatus != TriangulatorStatus::OK && track.triangulateStatus != TriangulatorStatus::HYBRID) {
        visualizeTrack(colorFrame, trackTracker, DARK_RED);
        return;
    }
    visualizeTrack(colorFrame, trackTracker, WHITE);

    if (track.triangulateStatus == TriangulatorStatus::HYBRID) {
        visualizeTrack(colorFrame, trackProjection, GREEN);
    }
    else if (track.prepareVuStatus != PREPARE_VU_OK) {
        visualizeTrack(colorFrame, trackProjection, MAGENTA);
    }
    else if (!track.visualUpdateSuccess) {
        visualizeTrack(colorFrame, trackProjection, BLUE);
    }
    else if (track.blacklisted) {
        visualizeTrack(colorFrame, trackProjection, PURPLE);
    }
    else {
        // Success.
        visualizeTrack(colorFrame, trackProjection, BLACK);
    }

    // visualizeMatchingTracks(colorFrame, track.trackTracker, track.trackProjection, cv::Scalar(0, 0, 255, 255));
}

// Scale number from interval [min, max] to interval [0, 1], clamping
// if the input is outside [min, max].
static double scaleClamp(double s, double min, double max) {
    assert(min < max);
    if (s < min) s = min;
    if (s > max) s = max;
    return (s - min) / (max - min);
}

void visualizeCovariance(cv::Mat& frame, const Eigen::MatrixXd& P0, int mapPointOffset, bool correlation) {
    int n = static_cast<int>(P0.rows());

    Eigen::MatrixXd P = P0;

    int trail_size = 4;
    int max_vis_trail = std::min(n, 19 + trail_size * 7);

    int hybrid_map_size = 3;
    hybrid_map_size = std::min(hybrid_map_size, (n - mapPointOffset) / 3);

    int hyb_sz = hybrid_map_size*3;
    P.block(max_vis_trail, 0, hyb_sz, n) = P.block(mapPointOffset, 0, hyb_sz, n).eval();
    P.block(0, max_vis_trail, n, hyb_sz) = P.block(0, mapPointOffset, n, hyb_sz).eval();
    P.conservativeResize(max_vis_trail + hyb_sz, max_vis_trail + hyb_sz);

    n = P.rows();

    if (correlation) {
        Eigen::MatrixXd Q = util::cov2corr(P);

        // Draw the correlation dots.
        frame = cv::Mat(n, n, CV_32F, cv::Scalar(0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Darker means more correlation.
                frame.at<float>(i, j) = 1.0 - std::abs(Q(i, j));
            }
        }
    } else {
        frame = cv::Mat(n, n, CV_32F, cv::Scalar(0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                frame.at<float>(i, j) = 1.0 - scaleClamp(std::log10(std::abs(P(i, j))), -12.0, -1.0);
            }
        }
    }
    visualizeState(frame, 10.0, true, max_vis_trail);
}

void visualizeState(cv::Mat& frame, double scale, bool draw_state_lines, int hybrid_index) {
    int n = frame.rows;
    int m = frame.cols;

    // TODO Comparing gray values visually is difficult, should map the colors to some colorful gradient here.

    // Scale dots into squares.
    cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_NEAREST);

    // Draw separator lines.
    if (!draw_state_lines) return;
    assert(m == n);
    cv::Scalar color(0.0);
    std::vector<int> marks = {POS, VEL, ORI, BGA, BAA, BAT};
    int k = 0;
    while (CAM + k * 7 < hybrid_index) {
        marks.push_back(CAM + k * 7);
        k++;
    }
    k = 0;
    while (hybrid_index + k * 3 < n) {
        marks.push_back(hybrid_index + k * 3);
        k++;
    }
    for (auto mark : marks) {
        double m = scale * static_cast<double>(mark);
        double end = scale * static_cast<double>(n);
        cv::line(frame, cv::Point2d(0.0, m), cv::Point2d(end, m), color);
        cv::line(frame, cv::Point2d(m, 0.0), cv::Point2d(m, end), color);
    }
}
} // namespace odometry
