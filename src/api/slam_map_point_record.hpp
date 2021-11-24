#ifndef SLAM_MAP_POINT_RECORD_HPP
#define SLAM_MAP_POINT_RECORD_HPP

#include <Eigen/Dense>

namespace slam {

// This is for recording *all* the map points from a session for visualization purposes.
// The VIO API also has its own point cloud output which gives the "current" cloud.
class MapPointRecord {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class Type {
        ODOMETRY,
        SLAM,
    };

    struct Position {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        float t;
        Eigen::Vector3f p;
    };

    MapPointRecord(
        float t,
        const Eigen::Vector3f &p,
        const Eigen::Vector3f &normal,
        Type type
    )
    : removed(false), normal(normal), type(type)
    {
        positions.push_back(MapPointRecord::Position { .t = t, .p = p });
    }

    MapPointRecord() : removed(false) {}

    // Record of changes to the points, eg due to recurring triangulation.
    std::vector<Position> positions;
    // Set true if the map point was removed from the session, in which case the
    // last item of `positions` gives the time of removal.
    bool removed;
    // Viewing direction
    Eigen::Vector3f normal;
    Type type;
};

} // namespace slam

#endif
