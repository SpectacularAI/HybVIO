#include "draw_gl.hpp"

#include "../util/logging.hpp"

namespace draw {

RGBA methodColor(api::PoseHistory kind, const Theme &theme) {
    using P = api::PoseHistory;
    switch (kind) {
        case P::ARKIT: return theme.cyan;
        case P::ARCORE: return theme.yellow;
        case P::ARENGINE: return theme.magenta;
        case P::REALSENSE: return theme.green;
        case P::EXTERNAL: return theme.orange;
        case P::GROUND_TRUTH: return theme.fgEmph;
        case P::GPS: return theme.red;
        case P::OUR:
        case P::OUR_PREVIOUS: return theme.blue;
        default: break;
    }
    return theme.fg;
}

std::string methodName(api::PoseHistory kind) {
    using P = api::PoseHistory;
    switch (kind) {
        case P::ARKIT: return "ARKit";
        case P::ARCORE: return "ARCore";
        case P::ARENGINE: return "AR Engine";
        case P::REALSENSE: return "Intel RealSense T265";
        case P::EXTERNAL: return "External";
        case P::GROUND_TRUTH: return "Ground truth";
        case P::GPS: return "GPS";
        case P::OUR:
        case P::OUR_PREVIOUS: return "Spectacular AI (ours)";
        default: break;
    }
    return "?";
}

void horizontalGrid(double scale) {
    glLineWidth(2);

    glBegin(GL_LINES);

    constexpr float z = 0.0;
    float interval_ratio = scale * 1.5e-3;
    float grid_min = -1000.0f * interval_ratio;
    float grid_max = 1000.0f * interval_ratio;

    for (int x = -20; x <= 20; x += 1) {
        line(x * 50.0f * interval_ratio, grid_min, z, x * 50.0f * interval_ratio, grid_max, z);
    }
    for (int y = -20; y <= 20; y += 1) {
        line(grid_min, y * 50.0f * interval_ratio, z, grid_max, y * 50.0f * interval_ratio, z);
    }
    glEnd();
}

void camera(const Eigen::Matrix4f &T, float w) {
    const float h = w * 0.667f;
    const float z = h * 0.9f;
    glPushMatrix();
    glMultMatrixf(T.transpose().data());
    glBegin(GL_LINES);

    //draw::line(0.0f, 0.0f, 0.0f, w, h, z);
    //draw::line(0.0f, 0.0f, 0.0f, w, -h, z);
    draw::line(0.0f, 0.0f, 0.0f, -w, -h, z);
    //draw::line(0.0f, 0.0f, 0.0f, -w, h, z);

    draw::line(w, h, z, w, -h, z);
    draw::line(-w, h, z, -w, -h, z);
    draw::line(-w, h, z, w, h, z);
    draw::line(-w, -h, z, w, -h, z);

    glEnd();
    glPopMatrix();
}

void cameraPointCloud(
    const Eigen::Matrix4f &camToWorld,
    const vecVector3f &pointCloud,
    const vecVector3f &pointCloudColor,
    float pointAlpha)
{
    glBegin(GL_POINTS);
    Eigen::Vector4f color(0, 0, 0, pointAlpha);
    for (size_t i = 0; i < pointCloud.size(); ++i) {
        const Eigen::Vector3f vertex = (camToWorld * pointCloud.at(i).homogeneous()).hnormalized();
        glVertex3fv(vertex.data());
        if (!pointCloudColor.empty()) {
            color.segment<3>(0) = pointCloudColor.at(i);
            glColor4fv(color.data());
        }
    }
    glEnd();
}

void center(float size, const Theme &theme) {
    float w = 0.15f * size;
    float a = 0.8f * size;

    glPushMatrix();

    glLineWidth(3);

    glBegin(GL_LINE_STRIP);

    glColor3fv(theme.cyan.data());
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(size, 0.0f, 0.0f);
    glVertex3f(a, w, 0.0f);
    glVertex3f(a, -w, 0.0f);
    glVertex3f(size, 0.0f, 0.0f);
    glVertex3f(a, 0.0f, w);
    glVertex3f(a, 0.0f, -w);
    glVertex3f(size, 0.0f, 0.0f);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, size, 0.0f);
    glVertex3f(0.0f, a, w);
    glVertex3f(0.0f, a, -w);
    glVertex3f(0.0f, size, 0.0f);
    glVertex3f(w, a, 0.0f);
    glVertex3f(-w, a, 0.0f);
    glVertex3f(0.0f, size, 0.0f);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, size);
    glVertex3f(w, 0.0f, a);
    glVertex3f(-w, 0.0f, a);
    glVertex3f(0.0f, 0.0f, size);
    glVertex3f(0.0f, w, a);
    glVertex3f(0.0f, -w, a);
    glVertex3f(0.0f, 0.0f, size);

    glEnd();
    glPopMatrix();
}

Animation::Animation() {}

void Animation::handle() {
    auto t = std::chrono::steady_clock::now();
    auto it = fadeAways.begin();
    while (it != fadeAways.end()) {
        FadeAway &f = *it;
        double dur = (t - f.started).count() / 1e9;
        if (dur > f.showDuration) {
            fadeAways.erase(it);
            continue;
        }
        RGBA color = f.color;
        double alpha = 1.0 - dur / f.showDuration;
        if (alpha < 0.0) alpha = 0.0;
        if (alpha > 1.0) alpha = 1.0;
        color[3] = alpha;
        glColor4fv(color.data());
        draw::horizontalCircle(f.position, f.radius);
        ++it;
    }
}

void Animation::fadeAway(Eigen::Vector3f position, double radius, RGBA color) {
    fadeAways.push_back(FadeAway {
        .started = std::chrono::steady_clock::now(),
        .showDuration = 7.0,
        .color = color,
        .position = position,
        .radius = radius,
    });
}

} // namespace draw
