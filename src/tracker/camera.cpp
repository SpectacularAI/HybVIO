#include <cmath>
#include <array>
#include <sstream>
#include <iomanip>
#include "camera.hpp"
#include "../odometry/parameters.hpp"
#include "../util/logging.hpp"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace tracker {

namespace {

/**
 * A helper class with common methods for all supported camera models.
 * In practice, principal point and focal length are used to "normalize"
 * the pixel coordinates in both pihole and fisheye models
 */
class CameraBase : public Camera {
protected:
    Matrix3d cameraMatrix, inverseCameraMatrix;
    CameraBase(const api::CameraParameters &intrinsic) {
        double fx = intrinsic.focalLengthX;
        double fy = intrinsic.focalLengthY;
        double ppx = intrinsic.principalPointX;
        double ppy = intrinsic.principalPointY;
        cameraMatrix <<
            fx, 0, ppx,
            0, fy, ppy,
            0, 0, 1;
        inverseCameraMatrix = cameraMatrix.inverse();
    }

    double getFocalLength() const final {
        return (cameraMatrix(0, 0) + cameraMatrix(1, 1)) * 0.5;
    }

    api::CameraParameters getIntrinsic() const final {
        api::CameraParameters intrinsic;
        intrinsic.focalLengthX = cameraMatrix(0, 0);
        intrinsic.focalLengthY = cameraMatrix(1, 1);
        intrinsic.principalPointX = cameraMatrix(0, 2);
        intrinsic.principalPointY = cameraMatrix(1, 2);
        return intrinsic;
    }

    template <class List> void serializeToStream(const std::string &type, const List &coeff, std::ostream &os) const {
        os << type << ' '
            << std::setprecision(16)
            << cameraMatrix(0, 0) << ' ' // fx
            << cameraMatrix(1, 1) << ' ' // fy
            << cameraMatrix(0, 2) << ' ' // ppx
            << cameraMatrix(1, 2) << ' ' // ppy
            << coeff.size();
        for (double c : coeff) os << ' ' << c;
    }

    // assuming a variable pix that should be mapped to pixNorm
    std::string normalizePixelGlsl() const {
        std::ostringstream oss;
        oss << "const vec2 ifocal = vec2("
            << (1.0 / cameraMatrix(0, 0)) << ","
            << (1.0 / cameraMatrix(1, 1)) << ");\n"
            << "const vec2 pp = vec2("
            << cameraMatrix(0, 2) << ","
            << cameraMatrix(1, 2) << ");\n"
            << "vec2 pixNorm = (pix - pp) * ifocal;\n";
        return oss.str();
    }

    // assuming a variable pixNorm that should be mapped to pix
    std::string unnormalizePixelGlsl() const {
        std::ostringstream oss;
        oss << "const vec2 focal = vec2("
            << cameraMatrix(0, 0) << ","
            << cameraMatrix(1, 1) << ");\n"
            << "const vec2 pp = vec2("
            << cameraMatrix(0, 2) << ","
            << cameraMatrix(1, 2) << ");\n"
            << "vec2 pix = pixNorm * focal + pp;\n";
        return oss.str();
    }
};

class PinholeCamera : public CameraBase {
private:
    static constexpr std::size_t N_COEFFS = 3;
    static constexpr int MAX_ITER = 100;
    static constexpr double EPS = 1e-5;
    bool distortionEnabled;
    const std::vector<double> coeffs;
    int width, height;
    bool rotationEnabled;
    Eigen::Matrix3d rotation;

    void distort(double &x, double &y, Eigen::Matrix2d* dDist) const {
        /*
         * As for distortion model, we use radial distortion model from opencv (without tangential distortion)
         * https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
         * We only use k1, k2, k3 parameters.
         */
        if (!distortionEnabled) {
            dDist->setIdentity();
            return;
        }

        double r2 = x * x + y * y;
        double theta = 1 + r2 * (coeffs[0] + r2 * (coeffs[1] + r2 * coeffs[2]));
        double dtheta = coeffs[0] + r2 * (coeffs[1] * 2 + r2 * coeffs[2] * 3);// d theta / d r^2
        *dDist <<
            theta + x * dtheta * 2 * x, x * dtheta * 2 * y,
            y * dtheta * 2 * x, theta + y * dtheta * 2 * y;
        x = x * theta;
        y = y * theta;
    }

    Vector2d undistort(const Vector2d& distortedPoint) const {
        if (!distortionEnabled) {
            return distortedPoint;
        }

        Vector2d point = distortedPoint;
        Eigen::Matrix2d J;
        Eigen::Vector2d delta;
        int iter = 0;
        do {
            double x = point.x(), y = point.y();
            distort(x, y, &J);
            delta = J.inverse() * (distortedPoint - Eigen::Vector2d{x, y});
            point += delta;
        } while (delta.norm() > EPS && ++iter < MAX_ITER);
        return point;
    }

    static bool isRotated(const Eigen::Matrix3d &rot) {
        return (rot - Eigen::Matrix3d::Identity()).norm() > 1e-8;
    }

    static std::string matToGlsl(const Eigen::Matrix3d &rot) {
        std::stringstream oss;
        oss << "mat3(";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i > 0 || j > 0) oss << ", ";
                oss << rot(j, i); // note: column major in GLSL
            }
        }
        oss << ")";
        return oss.str();
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PinholeCamera(const api::CameraParameters &intrinsic, const std::vector<double> &_coeffs, int w, int h, const Eigen::Matrix3d *rotPtr = nullptr)
    :
        CameraBase(intrinsic), coeffs(_coeffs), width(w), height(h),
        rotationEnabled(rotPtr && isRotated(*rotPtr)),
        rotation(rotationEnabled ? *rotPtr : Eigen::Matrix3d::Identity())
    {
        if (_coeffs.empty() || (_coeffs.size() == 1 && _coeffs[0] == 0.)) {
            distortionEnabled = false;
        } else {
            assert(_coeffs.size() == N_COEFFS);
            distortionEnabled = true;
        }

    }

    bool pixelToRay(const Vector2d &pixel, Vector3d &ray) const final {
        Vector2d point = {
                (pixel.x() - cameraMatrix(0, 2)) / cameraMatrix(0, 0),
                (pixel.y() - cameraMatrix(1, 2)) / cameraMatrix(1, 1)
        };
        point = undistort(point);
        ray = (Vector3d{point.x(), point.y(), 1}).normalized();
        if (rotationEnabled) ray = rotation * ray;
        return true;
    }

    bool rayToPixel(const Vector3d &ray0, Vector2d &pixel, Matrix2x3 *jacobian) const final {
        Eigen::Vector3d ray = ray0;
        if (rotationEnabled) ray = rotation.transpose() * ray;
        if (ray.z() <= 0) return false;
        double iz = 1.0 / ray.z();
        Eigen::Matrix2d dDist;
        Vector3d homog = ray * iz;
        distort(homog.x(), homog.y(), &dDist);
        const Vector3d uvn = cameraMatrix * homog;
        pixel << uvn.x(), uvn.y();

        if (jacobian != nullptr) {
            Eigen::Matrix<double, 2, 3> dHomog;
            dHomog <<
                iz, 0, -ray.x() * iz*iz,
                0, iz, -ray.y() * iz*iz;
            *jacobian = (cameraMatrix.block<2, 2>(0, 0) * dDist * dHomog);
            if (rotationEnabled) *jacobian = *jacobian * rotation.transpose();
        }
        return true;
   }

   bool isValidPixel(const Eigen::Vector2d &pixel) const final {
       if (width < 0 || height < 0) return true;
       // should check this... rather floor?
       int x = std::round(pixel.x()), y = std::round(pixel.y());
       return x >= 0 && x < width && y >= 0 && y < height;
   }

   std::string serialize() const final {
       std::ostringstream oss;
       serializeToStream("pinhole", coeffs, oss);
       oss << ' ' << width << ' ' << height;
       return oss.str();
   }

   std::string pixelToRayGlsl() const final {
        assert(!distortionEnabled);
        std::ostringstream oss;
        oss << "vec3 pixelToRay(vec2 pix) {\n"
            << normalizePixelGlsl() // declares "vec2 pixNorm"
            << "vec3 ray = normalize(vec3(pixNorm, 1.0));\n";
        if (rotationEnabled) {
            oss << "const mat3 rot = " << matToGlsl(rotation) << ";\n"
                << "ray = rot * ray;\n";
        }
        oss << "return ray;\n"
            << "}\n";
        return oss.str();
    }

    std::string rayToPixelGlsl() const final {
        std::ostringstream oss;
        oss << "vec2 rayToPixel(vec3 ray) {\n";
        if (rotationEnabled) {
            oss << "const mat3 rot = " << matToGlsl(rotation.transpose()) << ";\n"
                << "ray = rot * ray;\n";
        }
        oss << "vec2 pixNorm = ray.xy / ray.z;\n";
        if (distortionEnabled) {
            oss << "const vec3 coeffs = vec3("
                << coeffs[0] << ", "
                << coeffs[1] << ", "
                << coeffs[2] << ");\n";
            oss << R"(
            float r2 = dot(pixNorm, pixNorm);
            float theta = 1.0 + r2 * (coeffs.x + r2 * (coeffs.y + r2 * coeffs.z));
            pixNorm = pixNorm * theta;
            )";
        }
        oss << unnormalizePixelGlsl() // declares "vec2 pix"
            << "return pix;\n"
            << "}\n";

        return oss.str();
    }
};

class FisheyeCamera : public CameraBase {
private:
    static constexpr std::size_t N_COEFFS = 4;
    static constexpr double DISTORTION_TABLE_SIZE = 50;
    const bool distortionEnabled;
    std::array<double, N_COEFFS> distortionCoeffs;
    std::vector<double> undistortTable;
    const double maxValidThetaRad;
    double maxValidR;

    double distort(double theta, double *derivative) const {
        if (distortionEnabled) {
            const auto &k = distortionCoeffs;
            const double t = theta;
            const double t2 = t*t;
            if (derivative != nullptr) {
                *derivative = 1 + 3*t2*(k[0] + 5.0/3*t2*(k[1] + 7.0/5*t2*(k[2] + 9.0/7*t2*k[3])));
            }
            return t*(1 + t2*(k[0] + t2*(k[1] + t2*(k[2] + t2*k[3]))));
        } else {
            if (derivative != nullptr) *derivative = 1.0;
            return theta;
        }
    }

    // Newton's method. Should behave well with monotonic functions with
    // derivative close to 1.
    double undistortNewton(double r, double theta0) const {
        // Reaching the target accuracy should normally happen in about 3 iterations.
        constexpr std::size_t MAX_ITR = 20;
        constexpr double TARGET_EPS_PIXELS = 0.01; // approximate
        const double targetEps = TARGET_EPS_PIXELS / getFocalLength();

        double theta = theta0;
        double deltaTheta, dRdTheta;
        for (size_t itr = 0; itr < MAX_ITR; ++itr) {
            double deltaR = distort(theta, &dRdTheta) - r;
            deltaTheta = deltaR / dRdTheta;
            theta -= deltaTheta;

            if (std::abs(deltaTheta) < targetEps) {
                return std::max(theta, 0.0);
            }
        }
        return -1;
    }

    double undistort(double r) const {
        // Initialize optimization using a table. Seems to help Newton's method to converge faster.
        const size_t n = undistortTable.size();
        double theta = undistortTable.at(std::min(size_t(std::max(0.0, r / maxValidR) * n), n - 1));

        theta = undistortNewton(r, theta);
        if (theta < 0) {
            log_warn("undistort() did not converge.");
            return r;
        }
        return theta;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FisheyeCamera(const api::CameraParameters &intrinsic, const std::vector<double> &coeffs, double maxValidFovDeg = 180) :
        CameraBase(intrinsic),
        distortionEnabled(coeffs.size() > 1),
        distortionCoeffs(),
        undistortTable(),
        maxValidThetaRad(0.5 * maxValidFovDeg / 180.0 * M_PI)
    {
        if (distortionEnabled) {
            assert(coeffs.size() == N_COEFFS);
            for (std::size_t i = 0; i < N_COEFFS; ++i) distortionCoeffs[i] = coeffs[i];
        }

        maxValidR = distort(maxValidThetaRad, nullptr);

        if (distortionEnabled) {
            // Construct distortion lookup table.
            double theta = 0;
            const double step = maxValidR / DISTORTION_TABLE_SIZE;
            for (size_t i = 0; i < DISTORTION_TABLE_SIZE; ++i) {
                double r = (i + 0.5) * step;
                theta = undistortNewton(r, theta);
                assert(theta >= 0.0);
                undistortTable.push_back(theta);
                theta += step; // initial guess. could use the last derivative to improve
            }
        }
    }

    bool pixelToRay(const Vector2d &pixel, Vector3d &ray) const final {
        const Vector2d uv = (inverseCameraMatrix * Vector3d(pixel.x(), pixel.y(), 1)).segment<2>(0);
        const double r = uv.norm();
        const Vector2d dirXY = uv / r;

        bool success = true;
        double theta = r;
        if (r > maxValidR) {
            theta = maxValidThetaRad;
            success = false;
        }
        else if (distortionEnabled) {
            theta = undistort(r);
        }

        const double z = std::cos(theta);
        const Vector2d xy = std::sin(theta) * dirXY;
        ray = Vector3d(xy.x(), xy.y(), z);
        return success;
    }

    bool rayToPixel(const Vector3d &rayUnnorm, Vector2d &pixel, Matrix2x3 *jacobian) const final {
        if (rayUnnorm.z() <= 0) return false;
        const double invDist = 1.0 / rayUnnorm.norm();
        const Vector3d ray = rayUnnorm * invDist;

        // Equidistance projection
        const double theta = std::acos(ray.z());
        if (theta > maxValidThetaRad) return false;

        double dRdTheta;
        const double r = distort(theta, jacobian == nullptr ? nullptr : &dRdTheta);
        const Vector2d dirXY = rayUnnorm.segment<2>(0).normalized();
        const Vector2d uv = r * dirXY;
        const Vector3d homog(uv.x(), uv.y(), 1);
        pixel = (cameraMatrix * homog).segment<2>(0);

        if (jacobian != nullptr) {
            // gradient of ray.z()
            const Vector3d dRayZ = invDist * (Vector3d(0, 0, 1) - ray.z() * ray);
            // gradient of theta
            const Vector3d dTheta = -dRayZ / std::sqrt(1 - ray.z()*ray.z());
            // gradient of R
            const Vector3d dr = dRdTheta * dTheta;

            // Jacobian of dirXY
            using Eigen::Matrix2d;
            Matrix2x3 dDirXY = Matrix2x3::Zero();
            dDirXY.block<2,2>(0,0) = (Matrix2d::Identity() - dirXY * dirXY.transpose()) / rayUnnorm.segment<2>(0).norm();

            Matrix2x3 duv = dirXY * dr.transpose() + r * dDirXY;
            *jacobian = cameraMatrix.block<2, 2>(0, 0) * duv;
        }
        return true;
    }

    bool isValidPixel(const Eigen::Vector2d &pixel) const final {
        Eigen::Vector3d unused;
        return pixelToRay(pixel, unused);
    }

    std::string serialize() const final {
        std::ostringstream oss;
        serializeToStream("fisheye", distortionCoeffs, oss);
        oss << ' ' << (2 * maxValidThetaRad * 180 / M_PI); // maxValidFovDeg
        return oss.str();
    }

    std::string pixelToRayGlsl() const final {
        assert(!distortionEnabled);
        std::ostringstream oss;
        oss << "vec3 pixelToRay(vec2 pix) {\n"
            << normalizePixelGlsl() // declares "vec2 pixNorm"
            << R"(
            float r = length(pixNorm);
            vec2 dirXY = pixNorm / r;
            float theta = r;
            return vec3(
                sin(theta) * dirXY,
                cos(theta));
            }
            )";
        return oss.str();
    }

    std::string rayToPixelGlsl() const final {
        std::ostringstream oss;
        oss << "vec2 rayToPixel(vec3 rayUnnorm) {\n"
            << R"(
            float invDist = 1.0 / length(rayUnnorm);
            vec3 ray = rayUnnorm * invDist;
            float t = acos(ray.z);
            float r = t;
            )";

        if (distortionEnabled) {
            oss << "const vec4 k = vec4("
                << distortionCoeffs[0] << ", "
                << distortionCoeffs[1] << ", "
                << distortionCoeffs[2] << ", "
                << distortionCoeffs[3] << ");\n"
                << R"(
                float t2 = t * t;
                r = t*(1.0 + t2*(k.x + t2*(k.y + t2*(k.z + t2*k.w))));
                )";
        }

        oss << "vec2 pixNorm = r * normalize(ray.xy);\n"
            << unnormalizePixelGlsl() // declares "vec2 pix"
            << "return pix;\n"
            << "}\n";

        return oss.str();
    }
};

} // anonymous namespace

bool Camera::normalizePixel(const Eigen::Vector2d &pixel, Eigen::Vector2d &out) const {
   Eigen::Vector3d ray;
   if (!pixelToRay(pixel, ray) || ray.z() <= 0) return false;
   out = ray.segment<2>(0) / ray.z();
   return true;
}

std::unique_ptr<const Camera> Camera::buildPinhole(const api::CameraParameters &intrinsic) {
    std::vector<double> dummyCoeffs;
    return std::unique_ptr<const Camera>(new PinholeCamera(intrinsic, dummyCoeffs, -1, -1));
}

std::unique_ptr<const Camera> Camera::buildFisheye(const api::CameraParameters &intrinsic) {
    std::vector<double> dummyCoeffs;
    return std::unique_ptr<const Camera>(new FisheyeCamera(intrinsic, dummyCoeffs));
}

std::unique_ptr<const Camera> Camera::buildPinhole(
    const api::CameraParameters &intrinsic, const std::vector<double> &coeffs,
    int w, int h, const Eigen::Matrix3d *rotation)
{
    return std::unique_ptr<const Camera>(new PinholeCamera(intrinsic, coeffs, w, h, rotation));
}

std::unique_ptr<const Camera> Camera::buildFisheye(const api::CameraParameters &intrinsic, const std::vector<double> &coeffs, double maxValidFovDeg) {
    return std::unique_ptr<const Camera>(new FisheyeCamera(intrinsic, coeffs, maxValidFovDeg));
}

std::unique_ptr<const Camera> Camera::deserialize(const std::string &data) {
    // keepin' it simple
    std::istringstream iss(data);

    std::string type;
    api::CameraParameters intrinsic;
    int nCoeff;
    std::vector<double> coeff;
    iss >> type
        >> intrinsic.focalLengthX
        >> intrinsic.focalLengthY
        >> intrinsic.principalPointX
        >> intrinsic.principalPointY
        >> nCoeff;
    assert(nCoeff >= 0 && nCoeff <= 4);
    for (int i = 0; i < nCoeff; ++i) {
        double c;
        iss >> c;
        coeff.push_back(c);
    }
    if (type == "pinhole") {
        int w, h;
        iss >> w >> h;
        assert(iss);
        return Camera::buildPinhole(intrinsic, coeff, w, h);
    } else if (type == "fisheye") {
        double maxValidFovDeg;
        iss >> maxValidFovDeg;
        assert(iss);
        return Camera::buildFisheye(intrinsic, coeff, maxValidFovDeg);
    } else {
        log_error("invalid camera type %s", type.c_str());
        assert(false);
    }
    return {};
}

std::unique_ptr<const tracker::Camera> buildCamera(
    const api::CameraParameters &intrinsic,
    Camera::Kind kind,
    const odometry::ParametersTracker &pt,
    int width,
    int height,
    const std::vector<double> &coeffs
) {
    if (kind == Camera::Kind::FISHEYE) {
        return tracker::Camera::buildFisheye(intrinsic, coeffs, pt.validCameraFov);
    } else {
        assert(kind == Camera::Kind::PINHOLE);
        return tracker::Camera::buildPinhole(intrinsic, coeffs, width, height);
    }
}

}
