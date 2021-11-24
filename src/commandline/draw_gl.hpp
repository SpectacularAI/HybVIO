#ifndef DAZZLING_DRAW_GL_HPP
#define DAZZLING_DRAW_GL_HPP

#include "../api/internal.hpp" // For api::PoseHistory.
#include "../odometry/util.hpp" // vecVectorXf

#include <Eigen/Dense>
#include <pangolin/pangolin.h>

namespace draw {

using RGBA = std::array<float, 4>;

struct Theme {
    RGBA bg;
    RGBA bgEmph;
    RGBA secondary;
    RGBA fg;
    RGBA fgEmph;
    RGBA yellow;
    RGBA orange;
    RGBA red;
    RGBA magenta;
    RGBA violet;
    RGBA blue;
    RGBA cyan;
    RGBA green;
};

constexpr std::array<Theme, 3> themes = {{
    // <https://ethanschoonover.com/solarized/>
    // Solarized dark mode.
    {
        .bg        = {{    0 / 255.0,  43 / 255.0,  54 / 255.0, 1 }},
        .bgEmph    = {{    7 / 255.0,  54 / 255.0,  66 / 255.0, 1 }},
        .secondary = {{   88 / 255.0, 110 / 255.0, 117 / 255.0, 1 }},
        .fg        = {{  131 / 255.0, 148 / 255.0, 150 / 255.0, 1 }},
        .fgEmph    = {{  147 / 255.0, 161 / 255.0, 161 / 255.0, 1 }},
        .yellow    = {{  181 / 255.0, 137 / 255.0,   0 / 255.0, 1 }},
        .orange    = {{  203 / 255.0,  75 / 255.0,  22 / 255.0, 1 }},
        .red       = {{  220 / 255.0,  50 / 255.0,  47 / 255.0, 1 }},
        .magenta   = {{  211 / 255.0,  54 / 255.0, 130 / 255.0, 1 }},
        .violet    = {{  108 / 255.0, 113 / 255.0, 196 / 255.0, 1 }},
        .blue      = {{   38 / 255.0, 139 / 255.0, 210 / 255.0, 1 }},
        .cyan      = {{   42 / 255.0, 161 / 255.0, 152 / 255.0, 1 }},
        .green     = {{  133 / 255.0, 153 / 255.0,   0 / 255.0, 1 }},
    },
    // Solarized light mode.
    {
        .bg        = {{  253 / 255.0, 246 / 255.0, 227 / 255.0, 1 }},
        .bgEmph    = {{  238 / 255.0, 232 / 255.0, 213 / 255.0, 1 }},
        .secondary = {{  147 / 255.0, 161 / 255.0, 161 / 255.0, 1 }},
        .fg        = {{  101 / 255.0, 123 / 255.0, 131 / 255.0, 1 }},
        .fgEmph    = {{   88 / 255.0, 110 / 255.0, 117 / 255.0, 1 }},
        .yellow    = {{  181 / 255.0, 137 / 255.0,   0 / 255.0, 1 }},
        .orange    = {{  203 / 255.0,  75 / 255.0,  22 / 255.0, 1 }},
        .red       = {{  220 / 255.0,  50 / 255.0,  47 / 255.0, 1 }},
        .magenta   = {{  211 / 255.0,  54 / 255.0, 130 / 255.0, 1 }},
        .violet    = {{  108 / 255.0, 113 / 255.0, 196 / 255.0, 1 }},
        .blue      = {{   38 / 255.0, 139 / 255.0, 210 / 255.0, 1 }},
        .cyan      = {{   42 / 255.0, 161 / 255.0, 152 / 255.0, 1 }},
        .green     = {{  133 / 255.0, 153 / 255.0,   0 / 255.0, 1 }},
    },
    // Saturated light mode.
    {
        .bg        = {{   1,   1,   1, 1 }},
        .bgEmph    = {{ 0.8, 0.8, 0.8, 1 }},
        .secondary = {{ 0.4, 0.4, 0.4, 1 }},
        .fg        = {{ 0.2, 0.2, 0.2, 1 }},
        .fgEmph    = {{   0,   0,   0, 1 }},
        .yellow    = {{   1,   1,   0, 1 }},
        .orange    = {{   1, 0.6,   0, 1 }},
        .red       = {{   1,   0,   0, 1 }},
        .magenta   = {{   1,   0,   1, 1 }},
        .violet    = {{ 0.5,   0, 0.5, 1 }},
        .blue      = {{   0,   0,   1, 1 }},
        .cyan      = {{   0,   1,   1, 1 }},
        .green     = {{   0,   1,   0, 1 }},
    },
}};

class Animation {
public:
    Animation();

    void handle();
    void fadeAway(Eigen::Vector3f pos, double radius, RGBA color);

private:
    struct FadeAway {
        std::chrono::steady_clock::time_point started;
        double showDuration;
        draw::RGBA color;
        // Just circles for now.
        Eigen::Vector3f position;
        double radius;
    };

    std::vector<FadeAway> fadeAways;
};

inline void line(
    const float x1, const float y1, const float z1,
    const float x2, const float y2, const float z2
) {
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
}

inline void line(
    const Eigen::Vector3f &v1,
    const Eigen::Vector3f &v2
) {
    glVertex3f(v1(0), v1(1), v1(2));
    glVertex3f(v2(0), v2(1), v2(2));
}

inline void horizontalCircle(const Eigen::Vector3f &p, float r) {
    glBegin(GL_LINE_LOOP);
    for(float th = 0; th < 2 * M_PI; th += 0.3) {
        glVertex3f(p(0) + r * cos(th), p(1) + r * sin(th), p(2));
    }
    glEnd();
}

RGBA methodColor(api::PoseHistory kind, const Theme &theme);

std::string methodName(api::PoseHistory kind);

void horizontalGrid(double scale);

void camera(const Eigen::Matrix4f &T, float w);

void cameraPointCloud(
    const Eigen::Matrix4f &camToWorld,
    const vecVector3f &pointCloud,
    const vecVector3f &pointCloudColor,
    float pointAlpha = 1.0);

void center(float size, const Theme &theme);

} // namespace draw

#endif // DAZZLING_DRAW_GL_HPP
