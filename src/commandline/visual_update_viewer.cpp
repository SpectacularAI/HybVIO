#include "visual_update_viewer.hpp"

#include "../api/type_convert.hpp"
#include "../util/logging.hpp"
#include "../views/views.hpp"

#include "command_queue.hpp"
#include "codegen/output/cmd_parameters.hpp"
#include "../odometry/debug.hpp"
#include "../odometry/ekf.hpp"
#include "../odometry/ekf_state_index.hpp"
#include "../odometry/triangulation.hpp"
#include "../odometry/parameters.hpp"
#include "../odometry/util.hpp"

#include "draw_gl.hpp"
#include <pangolin/pangolin.h>
#include <array>
#include <mutex>

namespace odometry {
namespace viewer {

namespace {

constexpr double ALIGN_REFRESH_INTERVAL = 2.0;
constexpr bool USE_WAHBA = false;

// set to false for a normal depth test, which might look less confusing in
// certain scenarios
constexpr bool SHOW_GRID_BEHIND_EVERYTHING = true;

// associate point clouds with the first historical pose in the trail,
// which will not be IMU-predicted. Setting this to 0 will cause jitter
const size_t PC_EKF_IDX = 1;

void positionVertex(api::Vector3d v) {
    glVertex3d(v.x, v.y, v.z);
}
void setAlphaColor(draw::RGBA color, float alpha) {
    glColor4f(color[0], color[1], color[2], alpha);
}

vecMatrix4f extractTrail(
    const EKF &ekf,
    const Eigen::Matrix4d &imuToCamera
) {
    vecMatrix4f trail;
    trail.reserve(ekf.camTrailSize() + 1);
    for (int i = 0; i < ekf.camTrailSize() + 1; ++i) {
        const Eigen::Vector3d p = ekf.historyPosition(i - 1);
        const Eigen::Vector4d q = ekf.historyOrientation(i - 1);
        const Eigen::Matrix4d M = util::toWorldToCamera(p, q, imuToCamera);
        trail.push_back(M.cast<float>().inverse());
    }
    return trail;
}

void setRays(
    const std::vector<int> &poseTrailIndex,
    const vecVector2d &imageFeatures,
    std::vector<vecVector3f> &cameraPositions,
    std::vector<vecVector3f> &rays,
    vecMatrix4f &imuTrail,
    vecMatrix4f &camTrail,
    const EKF &ekf,
    const Parameters &parameters
) {
    {
        Eigen::Matrix4d imuToCamera = util::vec2matrix(parameters.odometry.imuToCameraMatrix);
        imuToCamera.block<3, 1>(0, 3) = Eigen::Vector3d::Zero();
        imuTrail = extractTrail(ekf, imuToCamera);
    }

    bool stereo = parameters.tracker.useStereo;
    size_t nCameras = stereo ? 2 : 1;
    rays.resize(nCameras);
    cameraPositions.resize(nCameras);
    std::vector<vecMatrix4f> trail;
    camTrail = extractTrail(ekf, parameters.imuToCamera);
    trail.push_back(camTrail);
    if (stereo) {
        trail.push_back(extractTrail(ekf, parameters.secondImuToCamera));
        for (const auto &matVec : trail.back()) camTrail.push_back(matVec);
    }

    assert(nCameras * poseTrailIndex.size() == imageFeatures.size());

    for (size_t camInd = 0; camInd < nCameras; ++camInd) {
        rays[camInd].reserve(poseTrailIndex.size());
        cameraPositions[camInd].reserve(poseTrailIndex.size());
        for (size_t n = 0; n < poseTrailIndex.size(); ++n) {
            const Eigen::Vector2d &ip = imageFeatures[camInd * poseTrailIndex.size() + n];
            int i = poseTrailIndex[n];
            Eigen::Vector3f p = trail[camInd][i].block<3, 1>(0, 3);
            Eigen::Vector3f v(ip(0), ip(1), 1.0);
            v.normalize();
            v = util::transformVec3ByMat4(trail[camInd][i], v) - p;
            v.normalize();
            rays[camInd].push_back(v);
            cameraPositions[camInd].push_back(p);
        }
    }
}

Eigen::Matrix4f apiToCamToWorld(const api::Pose &pose, const Eigen::Matrix4d &imuToCamera) {
    Eigen::Vector3d p = api::vectorToEigen(pose.position);
    Eigen::Vector4d q = api::quaternionToEigenVector(pose.orientation);
    if (q == Eigen::Vector4d::Zero()) q = Eigen::Vector4d(1, 0, 0, 0);
    return util::toWorldToCamera(p, q, imuToCamera).cast<float>().inverse();
}

struct DebugVisualUpdate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Some guess in case triangulation fails to estimate any point.
    float distance = 5.0;
    // Iterated triangulation results, last is most accurate.
    vecVector3f triangulationPoints;
    // Inner vectors have same length. Outer vector len = stereo ? 2 : 1.
    std::vector<vecVector3f> cameraPositions;
    std::vector<vecVector3f> rays;
    std::vector<int> poseTrailIndex;
    vecMatrix4f imuTrail, camTrail;
    // The inner vectors may be empty if visual update was aborted.
    std::vector<vecVector3f> afterCameraPositions;
    std::vector<vecVector3f> afterRays;
    vecMatrix4f afterImuTrail, afterCamTrail;
};

// for communication between DebugPublisher and VisualUpdateViewer
struct Data {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double age = 0.0;
    vecMatrix4f imuTrail, camTrail;
    vecMatrix4f positionTrack;
    std::vector<int> trailFrameNumbers;
    std::vector<api::Pose> poses;
    std::vector<DebugVisualUpdate> visualUpdates;
    Eigen::Vector3f accelerometer;
    Eigen::Vector3f gyroscope;
    vecVector3f pointCloud, pointCloudColor;

    struct PointCloud {
        vecVector3f positions, colors;
    };
    std::vector<std::shared_ptr<PointCloud>> trailPointClouds;
};

class DebugPublisherImpl : public DebugPublisher {
public:
    void startFrame(
        const EKF &ekf,
        const EKFStateIndex &ekfStateIndex,
        const Parameters &parameters
    ) final {
        Eigen::Matrix4d imuToCamera = util::vec2matrix(parameters.odometry.imuToCameraMatrix);
        imuToCamera.block<3, 1>(0, 3) = Eigen::Vector3d::Zero();
        vecMatrix4f imuTrail = extractTrail(ekf, imuToCamera);
        vecMatrix4f camTrail = extractTrail(ekf, parameters.imuToCamera);
        if (parameters.tracker.useStereo) {
            for (const auto &matVec : extractTrail(ekf, parameters.secondImuToCamera)) camTrail.push_back(matVec);
        }

        // Save current camera pose also in api::Pose format (for align()).
        const Eigen::Matrix4d T = odometry::util::toWorldToCamera(
                ekf.position(),
                ekf.orientation(),
                imuToCamera);
        Eigen::Vector3d position;
        Eigen::Vector4d orientation;
        odometry::util::toOdometryPose(T, position, orientation, Eigen::Matrix4d::Identity());
        api::Pose pose = api::eigenToPose(ekf.getPlatformTime(), position, orientation);

        std::lock_guard<std::mutex> l(m);
        data.imuTrail = imuTrail;
        data.camTrail = camTrail;
        data.positionTrack.push_back(imuTrail[0]);
        data.poses.push_back(pose);
        data.visualUpdates.clear();

        data.trailPointClouds.clear();
        data.trailFrameNumbers.clear();
        int minFrameNumber = ekfStateIndex.getFrameNumber(0);
        for (size_t i = 0; i < camTrail.size(); ++i) {
            std::shared_ptr<Data::PointCloud> pointCloud;
            int frameNumber = -1;
            if (i < ekfStateIndex.poseTrailSize()) {
                frameNumber = ekfStateIndex.getFrameNumber(i);
                if (pointCloudHistory.count(frameNumber)) {
                    pointCloud = pointCloudHistory.at(frameNumber);
                }
            }
            data.trailFrameNumbers.push_back(frameNumber);
            minFrameNumber = std::min(frameNumber, minFrameNumber);
            data.trailPointClouds.push_back(pointCloud);
        }
        assert(data.trailFrameNumbers.size() == camTrail.size());
        while (!pointCloudHistory.empty() && pointCloudHistory.begin()->first < minFrameNumber)
            pointCloudHistory.erase(pointCloudHistory.begin());
    }

    void startVisualUpdate(
        double age,
        const EKF &ekf,
        const std::vector<int> &poseTrailIndex,
        const vecVector2d &imageFeatures,
        const Parameters &parameters
    ) final {
        DebugVisualUpdate visualUpdate;
        visualUpdate.poseTrailIndex = poseTrailIndex;
        setRays(
            poseTrailIndex,
            imageFeatures,
            visualUpdate.cameraPositions,
            visualUpdate.rays,
            visualUpdate.imuTrail,
            visualUpdate.camTrail,
            ekf,
            parameters
        );

        std::lock_guard<std::mutex> l(m);
        data.visualUpdates.push_back(visualUpdate);
        data.age = age;
    }

    void finishSuccessfulVisualUpdate(
        const EKF &ekf,
        const std::vector<int> &poseTrailIndex,
        const vecVector2d &imageFeatures,
        const Parameters &parameters
    ) final {
        std::vector<vecVector3f> cameraPositions;
        std::vector<vecVector3f> rays;
        vecMatrix4f imuTrail, camTrail;
        setRays(poseTrailIndex, imageFeatures, cameraPositions, rays, imuTrail, camTrail, ekf, parameters);

        std::lock_guard<std::mutex> l(m);
        data.visualUpdates.back().afterRays = std::move(rays);
        data.visualUpdates.back().afterCameraPositions = std::move(cameraPositions);
        data.visualUpdates.back().afterCamTrail = std::move(camTrail);
        data.visualUpdates.back().afterImuTrail = std::move(imuTrail);
    }

    void pushTriangulationPoint(const Eigen::Vector3f &p) final {
        std::lock_guard<std::mutex> l(m);
        data.visualUpdates.back().triangulationPoints.push_back(p);
        data.visualUpdates.back().distance = (p - data.camTrail[0].block<3, 1>(0, 3)).norm();
    }

    void addSample(double t, const Eigen::Vector3f &g, const Eigen::Vector3f &a) final {
        aBuffer.col(imuInd) = a;
        gBuffer.col(imuInd) = g;
        imuInd = (imuInd + 1) % IMU_BUFFER_SIZE;
        if (t - tImuRefresh > IMU_REFRESH_INTERVAL || t < tImuRefresh) {
            tImuRefresh = t;

            std::lock_guard<std::mutex> l(m);
            data.accelerometer = aBuffer.rowwise().mean();
            data.gyroscope = gBuffer.rowwise().mean();
        }
    }

    void addPointCloud(const vecVector3f &pointsCamCoords, const vecVector3f *pointColors) {
        std::lock_guard<std::mutex> l(m);
        if (data.trailFrameNumbers.empty()) return;

        auto cloud = std::make_shared<Data::PointCloud>();
        cloud->positions = pointsCamCoords;
        if (pointColors) cloud->colors = *pointColors;

        assert(data.trailPointClouds.size() > PC_EKF_IDX);
        data.trailPointClouds.at(PC_EKF_IDX) = cloud;
        pointCloudHistory[data.trailFrameNumbers.at(PC_EKF_IDX)] = cloud;
    }

    void getData(Data &target) {
        std::lock_guard<std::mutex> l(m);
        target = data;
    }

private:
    static constexpr size_t IMU_BUFFER_SIZE = 100;
    static constexpr double IMU_REFRESH_INTERVAL = 0.02;

    std::mutex m;
    Data data;
    std::map<int, std::shared_ptr<Data::PointCloud>> pointCloudHistory;

    Eigen::Matrix<float, 3, IMU_BUFFER_SIZE> aBuffer;
    Eigen::Matrix<float, 3, IMU_BUFFER_SIZE> gBuffer;
    size_t imuInd = 0;
    double tImuRefresh = 0;
};

class VisualUpdateViewerImpl : public VisualUpdateViewer {
public:
    VisualUpdateViewerImpl(const cmd::Parameters &parameters, CommandQueue &commands) :
        commands(commands),
        imuToCamera(Eigen::Matrix4d::Identity()),
        theme(draw::themes[parameters.viewer.theme]),
        theme_ind(parameters.viewer.theme)
    {}

    DebugPublisher& getPublisher() final { return publisher; }

    void setFixedData(
        const PoseHistoryMap &poseHistoryMap,
        const Eigen::Matrix4d &imuToCamera,
        const Eigen::Matrix4d &secondImuToCamera
    ) final {
        (void)secondImuToCamera;
        this->poseHistoryMap = std::make_unique<PoseHistoryMap>(poseHistoryMap);
        this->imuToCamera = imuToCamera;
    }

    void setupFixedData() final {
        if (!poseHistoryMap) return;
        for (const auto &poseHistory : *poseHistoryMap) {
            PoseTrack poseTrack;
            poseTrack.poses = poseHistory.second;
            alignedPoses[poseHistory.first] = poseHistory.second;
            poseTracks[poseHistory.first] = std::move(poseTrack);
        }
        for (auto &poseTrack : poseTracks) {
            std::string name = draw::methodName(poseTrack.first);
            poseTrack.second.enabled = std::make_unique<pangolin::Var<bool>>("menu." + name, true, -1, 1, false);
        }
        poseHistoryMap.reset();
    }

    void setup() final {
        pangolin::CreateWindowAndBind(map_viewer_name_, viewerWidth, viewerHeight);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_DEPTH_TEST);

        float viewpoint_x_ = 0.0;
        float viewpoint_y_ = 0.1;
        float viewpoint_z_ = 15.0;
        float viewpoint_f_ = 600.0;
        s_cam = std::make_unique<pangolin::OpenGlRenderState>(
            pangolin::ProjectionMatrix(viewerWidth, viewerHeight, viewpoint_f_, viewpoint_f_,
                                       viewerWidth / 2, viewerHeight / 2, 0.1, 1e6),
            pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, 1.0, 0.0)
        );

        auto enforce_up = pangolin::AxisDirection::AxisZ;
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -viewerWidth / viewerHeight)
            .SetHandler(new pangolin::Handler3D(*s_cam, enforce_up));

        // Register all keyboard commands
        // Note! This doesn't conflict with cv::waitKey, only one of them can detect a single keypress, there is no duplication
        for (int key : commands.getKeys()) {
            pangolin::RegisterKeyPressCallback(key, [this, key]() {
                commands.keyboardInput(key);
            });
        }

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        menuShowTriangulation = std::make_unique<pangolin::Var<bool>>("menu.Show triangulation", true, -1, 1, false);
        menuShowImu = std::make_unique<pangolin::Var<bool>>("menu.Show IMU", false, -1, 1, false);
        menuAfterVu = std::make_unique<pangolin::Var<bool>>("menu.After visual update", false, -1, 1, false);
        menuShowGrid = std::make_unique<pangolin::Var<bool>>("menu.Show grid", true, true);
        menuStereoPc = std::make_unique<pangolin::Var<bool>>("menu.Stereo point cloud",  true, true);
        menuStereoPcTrail = std::make_unique<pangolin::Var<bool>>("menu.Point cloud trail",  true, true);
        menuChangeTheme = std::make_unique<pangolin::Var<bool>>("menu.Change theme", false, false);
        menuAlign = std::make_unique<pangolin::Var<bool>>("menu.Align tracks", true, true);
        menuMapScale = std::make_unique<pangolin::Var<float>>("menu.Map scale", 10.0, 2, 500, true);

        pangolin::RegisterKeyPressCallback('i', [&]() {
            cycle++;
            if (cycle >= static_cast<int>(data.visualUpdates.size())) cycle = -1;
        });
        pangolin::RegisterKeyPressCallback('u', [&]() {
            *menuAfterVu = !*menuAfterVu;
        });
    }

    void draw() final {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        assert(s_cam);
        d_cam.Activate(*s_cam);
        glClearColor(theme.bg.at(0), theme.bg.at(1), theme.bg.at(2), theme.bg.at(3));

        if (pangolin::Pushed(*menuChangeTheme)) {
            theme_ind = ++theme_ind % draw::themes.size();
            theme = draw::themes[theme_ind];
        }

        float scale = *menuMapScale;
        if (*menuShowGrid) {
            glColor3fv(theme.bgEmph.data());
            draw::horizontalGrid(scale);
            if (SHOW_GRID_BEHIND_EVERYTHING) glClear(GL_DEPTH_BUFFER_BIT);
            draw::center(0.2 * scale, theme);
        }

        setupFixedData();

        publisher.getData(data);
        if (data.age != currentAge) {
            currentAge = data.age;
            cycle = -1;
        }
        if (*menuAlign && (data.age >= alignAge + ALIGN_REFRESH_INTERVAL || data.age < alignAge)) {
            alignAge = data.age;
            PoseHistoryPtrMap poseHistoriesPtr;
            for (const auto &it : poseTracks) {
                // Copy to avoid repeated align of the same data that could cause loss of precision.
                alignedPoses[it.first] = it.second.poses;
                poseHistoriesPtr[it.first] = &alignedPoses[it.first];
            }
            poseHistoriesPtr[PoseHistory::OUR] = &data.poses; // Not modified by align().
            odometry::views::align(poseHistoriesPtr, PoseHistory::OUR, USE_WAHBA);
        }

        // Draw other method tracks.
        float cameraSize = 0.01 * scale;
        double time = data.poses.empty() ? 0.0 : data.poses.back().time;
        for (const auto &poseTrack : poseTracks) {
            if (!*poseTrack.second.enabled) continue;
            const std::vector<api::Pose> &poses = alignedPoses[poseTrack.first];
            if (poses.empty()) continue;
            size_t currentInd = 0;
            glBegin(GL_LINES);
            glColor3fv(draw::methodColor(poseTrack.first, theme).data());
            for (size_t i = 0; i + 1 < poses.size(); ++i) {
                positionVertex(poses[i    ].position);
                positionVertex(poses[i + 1].position);
                if (poses[i].time < time) currentInd = i;
            }
            glEnd();

            // NOTE `imuToCamera` only transforms the track correctly if the track is from the
            // same device that odometry is being run for.
            const Eigen::Matrix4f camToWorld = apiToCamToWorld(poses[currentInd], imuToCamera);
            draw::camera(camToWorld, cameraSize);
        }

        // Draw camera frustums for pose history.
        if (!*menuShowTriangulation || cycle == -1) {
            glColor3fv(draw::methodColor(PoseHistory::OUR, theme).data());
            for (const Eigen::Matrix4f &pose : data.camTrail) {
                draw::camera(pose, cameraSize);
            }
        }
        else if (cycle >= 0 && cycle < static_cast<int>(data.visualUpdates.size())) {
            glColor3fv(theme.red.data());
            const DebugVisualUpdate &u = data.visualUpdates[cycle];
            const vecMatrix4f &utrail = *menuAfterVu ? u.afterCamTrail : u.camTrail;
            if (!utrail.empty()) {
                for (int i : u.poseTrailIndex) {
                    assert(i < static_cast<int>(utrail.size()));
                    const Eigen::Matrix4f &pose = utrail[i];
                    draw::camera(pose, cameraSize);
                }
            }
        }

        if (*menuStereoPc) {
            glPointSize(1);
            const float pointAlpha = *menuStereoPcTrail ? 0.5 : 1.0;
            glColor4f(0.5, 0.5, 1, pointAlpha);
            for (size_t i = 0; i < data.camTrail.size(); ++i) {
                const auto pcPtr = data.trailPointClouds.at(i);
                if (!pcPtr) continue;
                draw::cameraPointCloud(data.camTrail.at(i),
                    pcPtr->positions, pcPtr->colors, pointAlpha);
                if (!*menuStereoPcTrail && !pcPtr->positions.empty()) break;
            }
        }

        // Draw line for position track.
        glColor3fv(draw::methodColor(PoseHistory::OUR, theme).data());
        glBegin(GL_LINES);
        for (size_t i = 0; i + 1 < data.positionTrack.size(); ++i) {
            glVertex3fv(data.positionTrack[i    ].block<3, 1>(0, 3).data());
            glVertex3fv(data.positionTrack[i + 1].block<3, 1>(0, 3).data());
        }
        glEnd();

        // Draw IMU samples rotated to world coordinates.
        // Gyroscope vector points down when turning clockwise around up axis, up when CCW.
        // Accelerometer vector points towards direction of acceleration, gravity ignored.
        if (*menuShowImu && !data.positionTrack.empty()) {
            Eigen::Vector3f p = data.positionTrack.back().block<3, 1>(0, 3);
            Eigen::Vector3f a = data.accelerometer - Eigen::Vector3f(0, 0, 9.81);
            glBegin(GL_LINES);
            glColor3fv(theme.red.data());
            draw::line(p, p + scale * 0.2 * a);
            glColor3fv(theme.green.data());
            draw::line(p, p + scale * 0.4 * data.gyroscope);
            glEnd();
        }

        if (*menuShowTriangulation) drawRays();

        pangolin::FinishFrame();
    }

private:
    struct PoseTrack {
        std::vector<api::Pose> poses;
        // Processed poses used by rendering functions.
        vecMatrix4f posesMat;
        draw::RGBA color;
        std::unique_ptr<pangolin::Var<bool>> enabled;
    };
    using PoseTrackMap = std::map<PoseHistory, PoseTrack>;

    void drawRays() {
        for (size_t i = 0; i < data.visualUpdates.size(); ++i) {
            if (cycle >= 0 && static_cast<int>(i) != cycle) continue;

            const DebugVisualUpdate &u = data.visualUpdates[i];
            // Make longer so that lines are still visible at the intersection.
            float distance = 2.0 * u.distance;

            // Rays.
            const std::vector<vecVector3f> &cameraPositions = *menuAfterVu ? u.afterCameraPositions : u.cameraPositions;
            const std::vector<vecVector3f> &rays = *menuAfterVu ? u.afterRays : u.rays;
            glBegin(GL_LINES);
            for (size_t camInd = 0; camInd < rays.size(); ++camInd) {
                draw::RGBA color = camInd == 0 ? theme.yellow : theme.violet;
                for (size_t j = 0; j < rays[camInd].size(); ++j) {
                    setAlphaColor(color, 0.0);
                    glVertex3fv((cameraPositions[camInd][j] + distance * rays[camInd][j]).eval().data());
                    setAlphaColor(color, 1.0);
                    glVertex3fv((cameraPositions[camInd][j]).data());
                }
            }

            // Lines between triangulation points.
            glColor3fv(theme.blue.data());
            for (size_t j = 0; j + 1 < u.triangulationPoints.size(); ++j) {
                glVertex3fv(u.triangulationPoints[j].data());
                glVertex3fv(u.triangulationPoints[j + 1].data());
            }
            glEnd();

            // Final triangulation point.
            glPointSize(10);
            glBegin(GL_POINTS);
            // Should show text on screen instead to indicate which mode is on.
            *menuAfterVu ? glColor3fv(theme.blue.data()) : glColor3fv(theme.red.data());
            if (!u.triangulationPoints.empty()) {
                glVertex3fv(u.triangulationPoints.back().data());
            }
            glEnd();
        }
    }

    DebugPublisherImpl publisher;
    CommandQueue &commands;
    double currentAge = 0.0;
    double alignAge = 0.0;
    Eigen::Matrix4d imuToCamera;

    // Changed with keyboard to choose what is shown in the window.
    int cycle = -1;

    Data data;
    std::unique_ptr<PoseHistoryMap> poseHistoryMap;
    PoseTrackMap poseTracks;
    PoseHistoryMap alignedPoses;

    draw::Theme theme;
    int theme_ind;

    pangolin::View d_cam;
    std::unique_ptr<pangolin::OpenGlRenderState> s_cam;

    std::unique_ptr<pangolin::Var<bool>> menuShowTriangulation;
    std::unique_ptr<pangolin::Var<bool>> menuShowImu;
    std::unique_ptr<pangolin::Var<bool>> menuAfterVu;
    std::unique_ptr<pangolin::Var<bool>> menuShowGrid;
    std::unique_ptr<pangolin::Var<bool>> menuStereoPc;
    std::unique_ptr<pangolin::Var<bool>> menuStereoPcTrail;
    std::unique_ptr<pangolin::Var<bool>> menuChangeTheme;
    std::unique_ptr<pangolin::Var<bool>> menuAlign;
    std::unique_ptr<pangolin::Var<float>> menuMapScale;

    const std::string map_viewer_name_{"Visual update viewer"};
    static constexpr float viewerWidth = 1024;
    static constexpr float viewerHeight = 768;
};
}

std::unique_ptr<VisualUpdateViewer> VisualUpdateViewer::create(const cmd::Parameters &p, CommandQueue &c) {
    return std::unique_ptr<VisualUpdateViewer>(new VisualUpdateViewerImpl(p, c));
}

VisualUpdateViewer::~VisualUpdateViewer() = default;

}
}
