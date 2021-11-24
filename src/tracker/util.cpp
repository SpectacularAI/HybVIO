#include "util.hpp"

#include "../util/util.hpp"
#include "../odometry/util.hpp"

#include <opencv2/opencv.hpp>
#include <memory>

namespace {
Eigen::Vector3d vec2eigen(const std::vector<double> &v) {
    assert(v.size() == 3);
    return Eigen::Vector3d(v.data());
}

// imageIn and imageOut can alias.
void setIntensities(
    const cv::Mat &imageIn,
    cv::Mat &imageOut,
    double mean,
    double stddev,
    double weight = 1.0
) {
    assert(imageIn.type() == CV_8UC1);
    assert(imageOut.type() == CV_8UC1);
    assert(imageIn.rows == imageOut.rows && imageIn.cols == imageOut.cols);
    assert(weight >= 0.0 && weight <= 1.0);

    cv::Scalar mean0scalar, stddev0scalar;
    cv::meanStdDev(imageIn, mean0scalar, stddev0scalar);
    double stddev0 = stddev0scalar.val[0];
    stddev = stddev0 + weight * (stddev - stddev0);

    assert(stddev0 > 0.0);
    uint64 imageInTotal = 0;
    for (int x = 0; x < imageIn.rows; ++x) {
        for (int y = 0; y < imageIn.cols; ++y) {
            imageInTotal += imageIn.at<uint8_t>(x, y);
        }
    }
    double devDivDev0 = stddev / stddev0;
    double mean0 = static_cast<double>(imageInTotal) * devDivDev0 / (imageIn.rows * imageIn.cols);
    mean = mean0 + weight * (mean - mean0);
    for (int x = 0; x < imageOut.rows; ++x) {
        for (int y = 0; y < imageOut.cols; ++y) {
            double v = static_cast<double>(imageIn.at<uint8_t>(x, y)) * devDivDev0 + mean - mean0;
            int u = static_cast<int>(round(v));
            if (u < 0) u = 0;
            if (u > 255) u = 255;
            imageOut.at<uint8_t>(x, y) = static_cast<uint8_t>(u);
        }
    }
}
}

namespace tracker {
namespace util {

// Prints elapsed time in milliseconds. Get the start time with:
//     int64_t since = cv::getTickCount();
void printTimeSince(const std::clock_t &since, const std::string &description) {
    log_debug("%s: %g ms", description.c_str(), timeSince(since));
}

double timeSince(const std::clock_t &since) {
    return 1000 * ((std::clock() - since) / double(CLOCKS_PER_SEC));
}

void automaticCameraParametersWhereUnset(
    odometry::Parameters &parameters
) {
    // NOTE Do not set principal point parameters here automatically, because
    // then they will override possible per-frame values. API uses image
    // mid-point as a fallback when neither handset parameter nor per-frame
    // value exists.

    odometry::ParametersTracker &tracker = parameters.tracker;
    odometry::ParametersOdometry &odometry = parameters.odometry;
    if (tracker.focalLengthX < 0) {
        tracker.focalLengthX = tracker.focalLength;
        tracker.focalLengthY = tracker.focalLength;
    }

    if (tracker.useStereo && tracker.secondFocalLengthX < 0) {
        if (tracker.secondFocalLength < 0) {
            log_debug("Second focal length unset, using first camera parameters for the second camera too");
            tracker.secondFocalLengthX = tracker.focalLengthX;
            tracker.secondFocalLengthY = tracker.focalLengthY;
            tracker.secondDistortionCoeffs = tracker.distortionCoeffs;
        } else {
            tracker.secondFocalLengthX = tracker.secondFocalLength;
            tracker.secondFocalLengthY = tracker.secondFocalLength;
        }
    }

    parameters.imuToCamera = odometry::util::vec2matrix(odometry.imuToCameraMatrix);
    if (odometry.secondImuToCameraMatrix.size() > 1) {
        parameters.secondImuToCamera = odometry::util::vec2matrix(odometry.secondImuToCameraMatrix);
    }
    else {
        parameters.secondImuToCamera = odometry::util::vec2matrix(odometry.imuToCameraMatrix);
    }

    if (odometry.secondImuToCameraMatrix.size() < 4 * 4) {
        parameters.secondImuToCamera.block<3, 1>(0, 3) += vec2eigen(odometry.stereoCameraTranslation);
    }

    if (odometry.outputCameraPose) {
        parameters.imuToOutput = parameters.imuToCamera;
    }

    // Currently we assume stereo frames have identical timestamps so we can only use
    // one value to shift them. However some datasets provide shift for each camera.
    if (odometry.secondImuToCameraShiftSeconds != 0.0) {
        if (odometry.imuToCameraShiftSeconds == 0.0) {
            // Unlikely combination.
            log_warn("Time shift provided only for second camera.");
        }
        odometry.imuToCameraShiftSeconds = 0.5 *
            (odometry.imuToCameraShiftSeconds + odometry.secondImuToCameraShiftSeconds);
    }
}

void scaleImageParameters(odometry::ParametersTracker& parameters, double videoScaleFactor) {
    // Focal length and principal point change with transformation
    // of the video.
#define X(FIELD) if (parameters.FIELD > 0.0) parameters.FIELD *= videoScaleFactor;
    X(focalLength)
    X(focalLengthX)
    X(focalLengthY)
    X(principalPointX)
    X(principalPointY)
    if (parameters.useStereo) {
        X(secondFocalLength)
        X(secondFocalLengthX)
        X(secondFocalLengthY)
        X(secondPrincipalPointX)
        X(secondPrincipalPointY)
    }
#undef X
}

// Rotate matrix a multiple of clockwise 90 degree rotations.
// Source and destination matrices can alias.
// Negative count rotates counterclockwise.
void rotateMatrixCW90(const cv::Mat &source, cv::Mat &dest, int times) {
    switch (::util::modulo(times, 4)) {
        case 1:
            cv::rotate(source, dest, cv::ROTATE_90_CLOCKWISE);
            break;
        case 2:
            cv::rotate(source, dest, cv::ROTATE_180);
            break;
        case 3:
            cv::rotate(source, dest, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            if (&source != &dest) source.copyTo(dest);
            break;
    }
}

// Turn two stereo images into grayscale and change second's values so that
// the mean and standard deviation of the intensities match. This was developed
// to fix issues with EuRoC data, in particular the set "Vicon Room 2 03".
//
// Comments on how to adapt this for GPU if necessary:
//
// 1. Downsample the images (there is a "convolution2D" operation, which could be applied with 1x1 kernel and large-ish stride. It works automatically for both CPU and GPU images)
// 2. Read the downsampled images to CPU as uint8_t arrays
// 3. Compute mean & stdev on the uint8_t arrays using the CPU
// 4. Compute the correction factors a, b for the linear correction x -> a*x + b for the right image (or both images if we also want to smooth out intensity changes for mono)
// 5. Apply the above tranformation with accelerated arrays (it's called "channelwise affine" transformation) there
void matchIntensities(cv::Mat &l, cv::Mat &r, double weight) {
    assert(l.type() == CV_8UC1);
    assert(r.type() == CV_8UC1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(l, mean, stddev);
    setIntensities(r, r, mean.val[0], stddev.val[0], weight);
}

} // namespace util
} // namespace tracker
