#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <complex>

#include "views.hpp"
#include "../api/internal.hpp"
#include "../api/type_convert.hpp"
#include "../odometry/util.hpp"

namespace {
constexpr bool SHOW_POINT_CLOUD = true;
constexpr bool SHOW_BIASES = true;

inline double pow2(double x) {
    return x*x;
}

double getElement(const api::Vector3d &v, int i) {
    return api::vectorToEigen(v)[i];
}

cv::Point2f toMapImage(const cv::Point2f& corner, double s, const Eigen::Vector3d& p, int ax1, int ax2) {
    // Reverse y-axis for image coordinates.
    return corner + s * cv::Point2f(p[ax1], -p[ax2]);
}

cv::Point2f toMapImage(const cv::Point2f& corner, double s, const api::Vector3d& p, int ax1, int ax2) {
    return toMapImage(corner, s, api::vectorToEigen(p), ax1, ax2);
}

static void drawCameraAngle(
        cv::Mat& poseFrame,
        const cv::Point2f p1,
        const api::Quaternion& q,
        const cv::Scalar& color,
        int ax1,
        int ax2,
        int lineThickness
    ) {
    const Eigen::Vector4d qn = Eigen::Vector4d(q.w, q.x, q.y, q.z).normalized();
    Eigen::Matrix3d R = odometry::util::quat2rmat(qn);
    Eigen::Vector3d cameraFwd = R.transpose() * Eigen::Vector3d(0,0,-1);
    constexpr float s = 30.0;
    // flip y to get image coords
    auto p2 = p1 + s * cv::Point2f(cameraFwd(ax1), -cameraFwd(ax2));
    cv::line(poseFrame, p1, p2, color, lineThickness);
}

static void drawCamera3d(
        cv::Mat& poseFrame,
        const cv::Point2f p0,
        const api::Quaternion& q,
        const cv::Scalar& color,
        int ax1,
        int ax2
    ) {
    const Eigen::Vector4d qn = Eigen::Vector4d(q.w, q.x, q.y, q.z).normalized();
    Eigen::Matrix3d R = odometry::util::quat2rmat(qn);

    constexpr float scale = 30.0;
    constexpr float deviceW = 0.5 * scale;
    constexpr float deviceH = 1.0 * scale;

    constexpr int nVertices = 4;
    const float vertices[nVertices][3] = {
        { 0.5 * deviceW, 0.5 * deviceH, 0 },
        { 0.5 * deviceW, -0.5 * deviceH, 0 },
        { -0.5 * deviceW, -0.5 * deviceH, 0 },
        { -0.5 * deviceW, 0.5 * deviceH, 0 },
    };

    for (int i = 0; i < nVertices; ++i) {
        const auto &cur = vertices[i];
        const auto &prev = vertices[i == 0 ? nVertices - 1 : i - 1];

        Eigen::Vector3d p1 = R.transpose() * Eigen::Vector3d(prev[0], prev[1], prev[2]);
        Eigen::Vector3d p2 = R.transpose() * Eigen::Vector3d(cur[0], cur[1], cur[2]);

        // flip y to get image coords
        cv::line(poseFrame,
            p0 + cv::Point2f(p1(ax1), -p1(ax2)),
            p0 + cv::Point2f(p2(ax1), -p2(ax2)),
            color);
    }
}

static void drawUncertainty(
        cv::Mat& poseFrame,
        const cv::Point2f p0,
        const api::Matrix3d &cov3d,
        float scale,
        int ax1,
        int ax2
    ) {

    // compute uncertainty ellipse from covariance using SVD
    Eigen::Matrix2d cov2d;
    cov2d <<
        cov3d[ax1][ax1], cov3d[ax1][ax2],
        cov3d[ax2][ax1], cov3d[ax2][ax2];
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov2d, Eigen::ComputeFullV);

    const Eigen::Vector2d sigma = svd.singularValues();
    const Eigen::Vector2d v1 = svd.matrixV().col(0);

    const cv::Scalar uncertaintyColor(100, 100, 100);

    // if the matrix is diagonal, then this is just
    // [sqrt(cov_xx), sqrt(cov_yy)] * scale
    const cv::Size axes(
        std::sqrt(sigma.x())*scale,
        std::sqrt(sigma.y())*scale);

    // minus sign because the image y-axis points down
    const double angle = -std::atan2(v1.y(), v1.x()) / M_PI * 180;

    // image, center, axes, angle, arc begin, arc end (deg), color
    cv::ellipse(poseFrame, p0, axes, angle, 0, 360, uncertaintyColor);
}

void visualizePoseWindow(
    odometry::views::PoseOverlayVisualization& pov,
    const PoseHistoryMap &poseHistories,
    cv::Mat &poseFrame,
    const api::VioApi::VioOutput &output,
    const std::map<int, api::Vector3d> &pointCloudHistory,
    int ax1, int ax2, bool centerCurrentPos,
    bool mobileVisualizations
) {
    using P = api::PoseHistory;
    bool someMethodShown = false;
    for (auto it = pov.methodOutputs.begin(); it != pov.methodOutputs.end(); ++it) {
        if (it->second.shown) someMethodShown = true;
    }
    if (!someMethodShown) return;

    int lineThickness = 1;
    if (mobileVisualizations) {
        lineThickness = 2;
    }
    double stickLength = 1.0; // In meters.
    constexpr double padding = 20.0; // In pixels.
    const double width = static_cast<double>(poseFrame.cols) - 2 * padding;
    const double height = static_cast<double>(poseFrame.rows) - 2 * padding;

    const std::vector<api::Pose> &poseHistoryOur = poseHistories.at(P::OUR);
    const std::vector<api::Pose> &poseHistoryARKit = poseHistories.at(P::ARKIT);

    // Compute scaling and translation.
    double currentTime = -1.0;
    double x0 = 0;
    double y0 = 0;
    double xlim[2] = { 0, 0 };
    double ylim[2] = { 0, 0 };
    if (!poseHistoryOur.empty()) {
        x0 = getElement(poseHistoryOur.back().position, ax1);
        y0 = getElement(poseHistoryOur.back().position, ax2);
        currentTime = poseHistoryOur.back().time;
    }
    else if (!poseHistoryARKit.empty()) {
        x0 = getElement(poseHistoryARKit.back().position, ax1);
        y0 = getElement(poseHistoryARKit.back().position, ax2);
        currentTime = poseHistoryARKit.back().time;
    }
    if (centerCurrentPos) {
        constexpr double WND_WIDTH_METERS = 5.0;
        xlim[0] = x0 - WND_WIDTH_METERS * 0.5;
        xlim[1] = x0 + WND_WIDTH_METERS * 0.5;
        ylim[0] = y0 - WND_WIDTH_METERS * 0.5;
        ylim[1] = y0 + WND_WIDTH_METERS * 0.5;
    }
    else {
        // Fit all position tracks into the view.
        bool first = true;
        for (const auto &it : pov.methodOutputs) {
            if (!it.second.shown) continue;
            for (const api::Pose &pose : poseHistories.at(it.first)) {
                const double p1 = getElement(pose.position, ax1);
                const double p2 = getElement(pose.position, ax2);
                if (first) {
                    xlim[0] = p1;
                    xlim[1] = p1;
                    ylim[0] = p2;
                    ylim[1] = p2;
                    first = false;
                }
                if (p1 < xlim[0]) { xlim[0] = p1; }
                if (p1 > xlim[1]) { xlim[1] = p1; }
                if (p2 > ylim[1]) { ylim[1] = p2; }
                if (p2 < ylim[0]) { ylim[0] = p2; }
            }
        }
        if (first) {
            return;
        }
        constexpr double padding_meters = 1.0;
        xlim[0] -= padding_meters;
        xlim[1] += padding_meters;
        ylim[0] -= padding_meters;
        ylim[1] += padding_meters;
        double h = std::max(xlim[1] - xlim[0], ylim[1] - ylim[0]);
        while (h > 11.0 * pov.stickLength) {
            pov.stickLength *= 10.0;
        }
        stickLength = pov.stickLength;
    }
    assert(xlim[1] > xlim[0] && ylim[1] > ylim[0]);
    double sx = width / (xlim[1] - xlim[0]);
    double sy = height / (ylim[1] - ylim[0]);

    if (sx <= 0.0 || sy <= 0.0) {
        return;
    }

    // Scale both axes by the most constraining factor.
    double s = std::min(sx, sy);
    double px = width - s * (xlim[1] - xlim[0]);
    double py = height - s * (ylim[1] - ylim[0]);
    // Leave `padding` worth of space in the more constrained direction and center
    // along the other dimension.
    cv::Point2f corner(
            -s * xlim[0] + px / 2 + padding,
            s * ylim[1] + py / 2 + padding // Reverse y-axis for image coordinates.
            );

    if (pov.methodOutputs[P::OUR].shown) {
        // Draw historical point cloud
        for (const auto &point : pointCloudHistory) {
            constexpr unsigned char bright = 100;
            cv::Scalar color(bright, bright, bright, 255);
            auto p = toMapImage(corner, s, point.second, ax1, ax2);
            cv::circle(poseFrame, p, 0, color);
        }

        // Draw camera pose trail coordinates and camera orientations.
        const auto &poseTrail = output.poseTrail;
        const int trailSize = static_cast<int>(poseTrail.size());
        cv::Scalar pathColor(0, 0, 255, 255);

        for (int i = 0; i < trailSize; i++) {
            auto p1 = toMapImage(corner, s, poseTrail[i].position, ax1, ax2);
            if (i + 1 < trailSize) {
                auto p2 = toMapImage(corner, s, poseTrail[i + 1].position, ax1, ax2);
                cv::line(poseFrame, p1, p2, pathColor);
            }

            const cv::Scalar cameraAngleColor(120, 120, 120, 255);
            drawCameraAngle(poseFrame, p1, poseTrail[i].orientation, cameraAngleColor, ax1, ax2, 1);
        }
    }

    // Draw method outputs.
    assert(pov.methodOutputs.size() >= 1);

    // Drawing lines on CPU is slow, so set minimum length of lines to draw.
    constexpr double LINE_SPACING_PIXELS = 4.0;
    const double spacing2 = pow2(LINE_SPACING_PIXELS);

    // reverse order so OURS (first in the enum) is drawn on top
    for (auto it = pov.methodOutputs.rbegin(); it != pov.methodOutputs.rend(); ++it) {
        if (!it->second.shown) continue;
        const std::vector<api::Pose> &methodPoses = poseHistories.at(it->first);
        cv::Scalar color = it->second.color;
        const size_t trailSize = methodPoses.size();
        if (methodPoses.empty()) {
            continue;
        }

        Eigen::Vector3d p1 = api::vectorToEigen(methodPoses[0].position);
        auto mp1 = toMapImage(corner, s, p1, ax1, ax2);
        for (size_t i = 1; i < trailSize; ++i) {
            Eigen::Vector3d p2 = api::vectorToEigen(methodPoses[i].position);
            auto mp2 = toMapImage(corner, s, p2, ax1, ax2);
            auto d = mp1 - mp2;
            if (d.x * d.x + d.y * d.y > spacing2) {
                cv::line(poseFrame, mp1, mp2, color, lineThickness);
                mp1 = mp2;
            }
        }

        // Draw circle at the current time position.
        if (it->first != api::PoseHistory::OUR) {
            int ind_gt = odometry::views::getIndexWithTime(methodPoses, currentTime);
            auto p = toMapImage(corner, s, methodPoses[ind_gt].position, ax1, ax2);
            drawCameraAngle(poseFrame, p, methodPoses[ind_gt].orientation, color, ax1, ax2, lineThickness);
            drawCamera3d(poseFrame, p, methodPoses[ind_gt].orientation, color, ax1, ax2);
        }

        // Draw legend in the big window.
        // Commented out because the rendering is quite slow.
        /*
        int legendPos = 0;
        if (!centerCurrentPos && ax1 == 0 && ax2 == 1) {
            cv::putText(
                    poseFrame,
                    it->second.legend,
                    cv::Point(20, poseFrame.rows - 20 * (legendPos + 1)),
                    cv::FONT_HERSHEY_PLAIN,
                    1.0,
                    color);
            legendPos++;
        }
        */
    }

    // Draw current camera orientation.
    if (pov.methodOutputs[P::OUR].shown && !poseHistoryOur.empty()) {
        auto p1 = toMapImage(corner, s, output.pose.position, ax1, ax2);
        const cv::Scalar cameraAngleColor(255, 255, 255, 255);
        const cv::Scalar camera3dColor(0, 0, 0, 255);
        drawUncertainty(poseFrame, p1, output.positionCovariance, s, ax1, ax2);
        drawCameraAngle(poseFrame, p1, output.pose.orientation, cameraAngleColor, ax1, ax2, lineThickness);
        drawCamera3d(poseFrame, p1, output.pose.orientation, camera3dColor, ax1, ax2);
    }

    // Draw the meter stick.
    {
        cv::Point2f rightMiddle(width, height / 2 + padding);
        cv::Point2f h1 = rightMiddle + cv::Point2f(0.0, -stickLength / 2 * s);
        cv::Point2f h2 = rightMiddle + cv::Point2f(0.0, stickLength / 2 * s);
        cv::Point2f side(10.0, 0.0);
        cv::Scalar color(0, 0, 0, 255);
        cv::line(poseFrame, h1, h2, color);
        cv::line(poseFrame, h1 - side, h1 + side, color);
        cv::line(poseFrame, h2 - side, h2 + side, color);
        int n = static_cast<int>(stickLength);
        if (n >= 10) {
            cv::putText(
                    poseFrame,
                    std::to_string(n),
                    rightMiddle,
                    cv::FONT_HERSHEY_PLAIN,
                    1.0,
                    color);
        }
    }

    // Draw feature points.
    if (pov.methodOutputs[P::OUR].shown && SHOW_POINT_CLOUD) {
        int sz = 2;
        for (auto& pf : output.pointCloud) {
            cv::Scalar color(0, 0, 0, 255);
            // hacky
            switch (pf.status) {
            case 0: // UNUSED
                color = cv::Scalar(128, 128, 128);
                break;
            case 2: // HYBRID
                color = cv::Scalar(0, 128, 0);
                break;
            case 3: // SLAM
                color = cv::Scalar(128, 190, 190);
                break;
            case 4: // OUTLIER
                color = cv::Scalar(0, 0, 128);
                break;
            case 5: // STEREO
                color = cv::Scalar(255, 0, 0);
                sz = 0;
                break;
            default: // POSE_TRAIL = 1
                break;
            }
            auto p = toMapImage(corner, s, pf.position, ax1, ax2);
            cv::circle(poseFrame, p, sz, color, 1);
        }
    }
}

// Nearest neighbor interpolation.
void interpolateGrid(
    const std::vector<api::Pose> &dataIn,
    const std::vector<api::Pose> &refIn,
    std::vector<Eigen::Vector3d> &dataOut,
    std::vector<Eigen::Vector3d> &refOut
) {
    assert(!dataIn.empty());
    assert(!refIn.empty());
    dataOut.clear();
    refOut.clear();
    dataOut.reserve(refIn.size());
    refOut.reserve(refIn.size());
    size_t i = 0;
    double tMin = std::max(dataIn[0].time, refIn[0].time);
    double tMax = std::max(dataIn.back().time, refIn.back().time);
    for (const api::Pose &r : refIn) {
        while (dataIn[i].time < r.time && i + 1 < dataIn.size()) ++i;
        if (r.time > tMin && r.time < tMax) {
            dataOut.push_back(api::vectorToEigen(dataIn[i].position));
            refOut.push_back(api::vectorToEigen(r.position));
        }
    }
    assert(dataOut.size() == refOut.size());
}

} // anonymous namespace

namespace odometry {
namespace views {

void visualizePose(
    PoseOverlayVisualization& pov,
    cv::Mat& poseFrame,
    const api::VioApi::VioOutput &output,
    const std::map<int, api::Vector3d>& pointCloudHistory,
    const PoseHistoryMap &poseHistoriesIn,
    bool mobileVisualizations,
    bool alignTracks
) {
    using P = api::PoseHistory;

    // Copy the pose histories so that we can transform them.
    PoseHistoryMap poseHistories = poseHistoriesIn;
    const std::vector<api::Pose> &posesOur = poseHistories.at(P::OUR);

    // Set starting time to zero.
    double t0 = -1.0;
    if (!posesOur.empty()) {
        t0 = posesOur[0].time;
    }
    else if (!poseHistories[P::ARKIT].empty()) {
        t0 = poseHistories[P::ARKIT][0].time;
    }

    for (auto it = poseHistories.begin(); it != poseHistories.end(); ++it) {
        std::vector<api::Pose> &poses = it->second;
        if (!poses.empty()) pov.methodOutputs.at(it->first).exists = true;
        for (size_t i = 0; i < poses.size(); ++i) {
            poses[i].time -= t0;
        }
    }
    // Some code assumes these exist.
    for (auto it = pov.methodOutputs.begin(); it != pov.methodOutputs.end(); ++it) {
        if (!poseHistories.count(it->first)) poseHistories[it->first] = {};
    }

    if (alignTracks) {
        // Skip align if the tracks are short, to avoid crazy spinning.
        const Eigen::Vector3d &p0 = api::vectorToEigen(posesOur[0].position);
        const Eigen::Vector3d &p1 = api::vectorToEigen(posesOur.back().position);
        if (pov.alignedBefore || (p1 - p0).squaredNorm() > 1.0) {
            pov.alignedBefore = true;
            align(poseHistories, P::OUR);
        }
    }

    if (poseFrame.dims == 0) {
        // if Mat is empty (e.g., cv::Mat())
        constexpr int defaultWidth = 500;
        constexpr int defaultHeight = 1000;
        constexpr int gray = 150;
        poseFrame = cv::Mat(defaultHeight, defaultWidth, CV_8UC4, cv::Scalar(gray, gray, gray, 255));
    }

    const int w = poseFrame.cols, h = poseFrame.rows;
    const int vSplit = 2*h/3;

    cv::Mat globalWindow = poseFrame(cv::Rect(0, 0, w, h/2));
    cv::Mat localWindow = poseFrame(cv::Rect(0, vSplit, w/2, h-vSplit));
    cv::Mat sideWindow = poseFrame(cv::Rect(w/2, vSplit, w/2, h-vSplit));

    visualizePoseWindow(pov, poseHistories, globalWindow, output, pointCloudHistory, 0, 1, false, mobileVisualizations);
    visualizePoseWindow(pov, poseHistories, localWindow, output, pointCloudHistory, 0, 1, true, mobileVisualizations);
    visualizePoseWindow(pov, poseHistories, sideWindow, output, pointCloudHistory, 0, 2, true, mobileVisualizations);

    // draw separators
    {
        const cv::Scalar sepColor(100, 100, 100, 255);
        cv::line(poseFrame, cv::Point2f(w/2, vSplit), cv::Point2f(w/2, h), sepColor);
        cv::line(poseFrame, cv::Point2f(0, vSplit), cv::Point2f(w, vSplit), sepColor);
    }

    if (!SHOW_BIASES) return;

    // Visualize biases
    struct BiasVisu {
        cv::Scalar color;
        float scale;
        api::Vector3d data;
        api::Vector3d covDiag;
        api::Vector3d reference;
    };

    int biasVisuIndex = 0;
    const auto &debug = reinterpret_cast<const api::InternalAPI::Output&>(output);
    const BiasVisu biasVisus[] = {
      {
        .color = cv::Scalar(255, 0, 0, 255),
        .scale = 1000,
        .data = debug.meanBAA,
        .covDiag = debug.covDiagBAA,
        .reference = { 0, 0, 0 }
      },
      {
        .color = cv::Scalar(0, 255, 0, 255),
        .scale = 10000,
        .data = debug.meanBAT,
        .covDiag = debug.covDiagBAT,
        .reference = { 1, 1, 1 }
      },
      {
        .color = cv::Scalar(0, 0, 255, 255),
        .scale = 1000,
        .data = debug.meanBGA,
        .covDiag = debug.covDiagBGA,
        .reference = { 0, 0, 0 }
      }
    };

    int lineThickness = 1;
    if (mobileVisualizations) {
        lineThickness = 2;
    }
    for (const auto &visu : biasVisus) {
        constexpr double corner = 50;
        cv::Point2f topLeft(corner, corner + biasVisuIndex*50);
        const auto v = (api::vectorToEigen(visu.data) - api::vectorToEigen(visu.reference)) * visu.scale;
        cv::Point2f vecXY = cv::Point2f(v(0), v(1));
        cv::Point2f vecXYZ = cv::Point2f(-0.5, -0.5) * v(2);
        const auto tip = topLeft + vecXY + vecXYZ;
        const auto shadow = topLeft + vecXY;
        cv::line(poseFrame, shadow, tip, cv::Scalar(130, 130, 130, 255), lineThickness);
        cv::line(poseFrame, topLeft, shadow, cv::Scalar(100, 100, 100, 255), lineThickness);
        cv::line(poseFrame, topLeft, tip, visu.color, lineThickness);

        const double uncertainty = std::sqrt(api::vectorToEigen(visu.covDiag).sum()) * visu.scale;
        cv::circle(poseFrame, tip, uncertainty, visu.color, 1, lineThickness);
        biasVisuIndex++;
    }
}

int getIndexWithTime(const std::vector<api::Pose>& poses, double t) {
    assert(poses.size() > 0);
    for (int j = 0; j < static_cast<int>(poses.size()); ++j) {
        if (poses[j].time >= t) {
            return j;
        }
    }
    return 0;
}

void align(PoseHistoryMap &poseHistories, api::PoseHistory ref, bool useWahba) {
    PoseHistoryPtrMap poseHistoriesPtr;
    for (auto &it : poseHistories) {
        poseHistoriesPtr[it.first] = &it.second;
    }
    align(poseHistoriesPtr, ref, useWahba);
}

void align(PoseHistoryPtrMap &poseHistories, api::PoseHistory ref, bool useWahba) {
    assert(poseHistories.find(ref) != poseHistories.end() && poseHistories.at(ref));
    const std::vector<api::Pose> &poseHistoryRef = *poseHistories.at(ref);
    if (poseHistoryRef.empty()) return;

    // Should skip histories that are not needed, eg those not shown in the `-p` visualization.
    for (auto it = poseHistories.begin(); it != poseHistories.end(); ++it) {
        if (it->first == ref) continue;

        assert(it->second);
        std::vector<api::Pose> &poseHistoryMethod = *it->second;
        if (poseHistoryMethod.empty()) continue;

        std::vector<Eigen::Vector3d> methodPositions;
        std::vector<Eigen::Vector3d> refPositions;
        interpolateGrid(poseHistoryMethod, poseHistoryRef, methodPositions, refPositions);
        assert(methodPositions.size() == refPositions.size());
        if (methodPositions.empty()) continue;
        Eigen::Vector3d methodMeanPosition = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d &p : methodPositions) {
            methodMeanPosition += p;
        }
        methodMeanPosition /= static_cast<double>(methodPositions.size());
        for (Eigen::Vector3d &p : methodPositions) {
            p -= methodMeanPosition;
        }
        // Cannot compute this outside the loop because it depends on the interpolation grid.
        Eigen::Vector3d refMeanPosition = Eigen::Vector3d::Zero();
        for (Eigen::Vector3d &p : refPositions) {
            refMeanPosition += p;
        }
        refMeanPosition /= static_cast<double>(refPositions.size());
        for (Eigen::Vector3d &p : refPositions) {
            p -= refMeanPosition;
        }

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (useWahba) {
            // <https://en.wikipedia.org/wiki/Wahba%27s_problem>
            // This gives full 3d rotation which generally works quite badly until the tracks
            // are long enough.
            Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < methodPositions.size(); ++i) {
                B += refPositions[i] * methodPositions[i].transpose();
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            // Might have some kind of mirroring issue. Check vio_benchmark's compute_metrics.py.
            R = U * V.transpose();
        }
        else {
            // Rotate around z-axis only.
            Eigen::ArrayXcd cr(methodPositions.size()), cm(methodPositions.size());
            for (size_t i = 0; i < methodPositions.size(); ++i) {
                cr(i) = std::complex<double>(refPositions[i](0), refPositions[i](1));
                cm(i) = std::complex<double>(methodPositions[i](0), methodPositions[i](1));
            }
            Eigen::ArrayXcd rot = cr / cm;
            rot /= rot.abs();
            if (!rot.hasNaN()) {
                R = Eigen::AngleAxisd(std::arg(rot.mean()), Eigen::Vector3d::UnitZ());
            }
        }

        for (api::Pose &pose : poseHistoryMethod) {
            Eigen::Vector3d p = api::vectorToEigen(pose.position) - methodMeanPosition;
            pose.position = api::eigenToVector(R * p + refMeanPosition);
            // The orientation `q` is world-to-IMU and so transforms as `q = R(q) * T.transpose()`.
            Eigen::Matrix3d M = util::quat2rmat(api::quaternionToEigenVector(pose.orientation));
            pose.orientation = api::eigenVectorToQuaternion(util::rmat2quat(M * R.transpose()));
        }
    }
}

}
}
