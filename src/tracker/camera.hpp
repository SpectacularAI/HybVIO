#ifndef TRACKER_CAMERA_H_
#define TRACKER_CAMERA_H_

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "../api/vio.hpp"

namespace odometry { struct ParametersTracker; }
namespace tracker {

class Camera {
public:
    enum class Kind {
        PINHOLE,
        FISHEYE,
    };

    using Matrix2x3 = Eigen::Matrix<double, 2, 3>; // for projection Jacobians
    virtual ~Camera() = default;

    /**
     * Pixel coordinates to camera ray
     *
     * @param pixel 2D pixel/image coordinates
     * @param ray the corresponding camera ray, i.e., the unit vector v pointing
     *      away from the camera such that all positive multiples of that ray
     *      (in the camera coordinate system) project to "pixel"
     * @return true if pixel lies in valid FOV of the camera. Note that the
     *      pixel does not need to necessarily be withing the image boundarides
     */
    virtual bool pixelToRay(const Eigen::Vector2d &pixel, Eigen::Vector3d &ray) const = 0;

    /**
     * Check if a pixel is valid. It must both be withing the valid FOV of
     * the camera (for fisheye-like projections) and withing the image
     * boundaries [0, width] x [0, height]. If the pixel boundaries have not
     * been set for some reason, then any point (withing the FOV) is valid.
     *
     * @param pixel input
     * @return output: true iff valid
     */
    virtual bool isValidPixel(const Eigen::Vector2d &pixel) const = 0;
    // shorthand for the above
    inline bool isValidPixel(double x, double y) const {
        return isValidPixel(Eigen::Vector2d(x, y));
    }

    /**
     * Convert pixel to a camera ray and compute the XY intersection of
     * that ray with the plane { z = 1 }. This is regularly used in, e.g.,
     * RANSAC5 code, but may be bad for general camera models
     */
    bool normalizePixel(const Eigen::Vector2d &pixel, Eigen::Vector2d &out) const;

    /**
     * Project a camera ray to its pixel coordinates and optionally compute the
     * Jacobian of this projection.
     *
     * @param ray (input) camera ray: unit vector pointing away from the camera
     * @param pixel (output) pixel coordinates corresponding to the ray
     * @param jacobian (output) if non-null, will be set to the Jacobian
     *      of the function ray -> pixel at this point
     * @return true if the projection is valid, that is, ray is in the range of
     *      pixelToRay. Otherwise false (for example, ray points behind the
     *      camera). Note: a true return value does not mean that the pixel is
     *      necessarily inside image boundaries. This has to be checked elsewhere
     *      and is not the reponsibility of the Camera model. If false, the values
     *      of both outputs (pixel & jacobian) are undefined
     */
    virtual bool rayToPixel(const Eigen::Vector3d &ray, Eigen::Vector2d &pixel, Matrix2x3 *jacobian = nullptr) const = 0;

    /**
     * @return GLSL implementation of `vec3 pixelToRay(vec2 pixel);`
     * without checks or derivatives, used for rectification
     */
    virtual std::string pixelToRayGlsl() const = 0;

    /**
     * @return GLSL implementation of `vec2 rayToPixel(vec3 ray);`
     * without checks or derivatives, used for rectification
     */
    virtual std::string rayToPixelGlsl() const = 0;

    /**
     * Focal length. For backwards compatilibity. TODO: remove this
     */
    virtual double getFocalLength() const = 0;

    /** Undistorted pinhole camera model */
    static std::unique_ptr<const Camera> buildPinhole(const api::CameraParameters &intrinsic);
    /**
     * Distorted pinhole camera model. Width and height are unused if negative
     * rotation unused if nullptr. Some equivalent interpretations of the
     * rotation matrix:
     *
     *  - extra camera-to-"world" matrix just in front of the camera
     *  - how the camera image plane is rotated, w.r.t. the base version
     *    where z corresponds to the principal axis
     *  - the rotation from "second camera" coordinates to "first camera"
     -    coordinates in stereo, if this is the "second camera"
     */
    static std::unique_ptr<const Camera> buildPinhole(
        const api::CameraParameters &intrinsic, const std::vector<double> &distortionCoefficients,
        int width = -1, int height = -1, const Eigen::Matrix3d *rotation = nullptr);

    /*
     * Undistorted fisheye camera (equidistance projection). Equivalent to the
     * distorted version with all distortion coefficients set to zero
     */
    static std::unique_ptr<const Camera> buildFisheye(const api::CameraParameters &intrinsic);
    /**
     * Distorted fisheye camera model, radially symmetric Kannala-Brandt
     * distortion model with 4 distortion coefficients
     * @param intrinsic Intrinsic parameters in pixel units
     * @param distortionCoefficients. Vector of length 0, 1 or 4. If the length
     *    is 0 or 1, this is interpreted to mean "no distortion" and the model
     *    is the same as the undistorted one, or equivalently, one with all-zero
     *    distortion coefficient { 0, 0, 0, 0 }. If the vector has 4 components
     *    (k1, k2, k3, k4), the, the angle t between the camera ray and the
     *    principal axis is mapped to the radial part of normalized image
     *    coordinates as
     *
     *      r = t + k1 * t^3 + k2 * t^5 + k3 * t^7 + k4 * t^9
     *
     *    Note that the coefficient of t can be 1 without loss of generality
     *    because, after this, the normalized image coorindates are mapped
     *    to pixel coordinates as
     *
     *      x = fx * cos(phi) * r + ppx
     *      y = fy * sin(phi) * r + ppy
     *
     *    This convention is used, e.g., in the RealSense API and is different
     *    from the one in the the original paper [1] where k1 multiplies t.
     *
     *    [1] http://www.ee.oulu.fi/~jkannala/calibration/Kannala_Brandt_calibration.pdf
     *
     * @param maxAngleDeg the field of view angle for which the distorted model
     *    is expected to be valid. Note that even if the fisheye model itself
     *    could be valid for even 360 FOV, this may not be true for all choices
     *    of the distortion coefficients. In particular, the distortion function
     *    may be non-monotonic outside its inteneded domain. Therefore it's good
     *    to state the valid domain for the angle explicitly.
     */
    static std::unique_ptr<const Camera> buildFisheye(
        const api::CameraParameters &intrinsic,
        const std::vector<double> &distortionCoefficients, double maxValidFovDeg = 180);

    virtual api::CameraParameters getIntrinsic() const = 0;

    // custom serialization (can be used with, e.g., cereal)
    virtual std::string serialize() const = 0;
    static std::unique_ptr<const Camera> deserialize(const std::string &data);
};

std::unique_ptr<const tracker::Camera> buildCamera(
    const api::CameraParameters &intrinsic,
    Camera::Kind kind,
    const odometry::ParametersTracker &pt,
    int width,
    int height,
    const std::vector<double> &coeffs
);

} // namespace tracker

#endif
