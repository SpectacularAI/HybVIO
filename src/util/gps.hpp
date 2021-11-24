#ifndef DAZZLING_API_GPS_UTIL_HPP
#define DAZZLING_API_GPS_UTIL_HPP

#include <Eigen/Dense>
#include <cmath>

namespace util {
/**
 * Convert from GPS coordinates to a local metric coordinate system:
 * East-North-Up (ENU) coordinates centered on the first position
 */
class GpsToLocalConverter {
private:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool initialized = false;
    double lat0, lon0, z0;
    double metersPerLonDeg;
    Eigen::Vector3d prev;

    inline static double degToRad(double a) {
        return a * M_PI / 180;
    }

public:
    GpsToLocalConverter() : prev(0, 0, 0) {}

    Eigen::Vector3d convert(
        double latitude,
        double longitude,
        double altitude = 0.0,
        double accuracy = 1.0,
        double minAccuracy = -1.0
    ) {
        // Filter out inaccurate measurements to make pose alignment easier.
        if (minAccuracy > 0.0 && (accuracy > minAccuracy || accuracy < 0.0)) {
            return prev;
        }

        constexpr double EARTH_CIRCUMFERENCE_EQUATORIAL = 40075.017e3;
        constexpr double EARTH_CIRCUMFERENCE_POLAR = 40007.863e3;
        constexpr double METERS_PER_LAT_DEG = EARTH_CIRCUMFERENCE_POLAR / (2 * M_PI);
        if (!initialized) {
            lat0 = latitude;
            lon0 = longitude;
            z0 = altitude;
            metersPerLonDeg = std::cos(degToRad(lat0)) * EARTH_CIRCUMFERENCE_EQUATORIAL / (2 * M_PI);
        }
        initialized = true;

        const Eigen::Vector3d vec(
            metersPerLonDeg * (degToRad(longitude - lon0)),
            METERS_PER_LAT_DEG * (degToRad(latitude - lat0)),
            altitude - z0);

        prev = vec;
        return vec;
    }
};
}

#endif
