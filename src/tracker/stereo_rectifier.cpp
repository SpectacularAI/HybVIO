#include "stereo_rectifier.hpp"

#include "../odometry/parameters.hpp"
#include "camera.hpp"

#include <opencv2/calib3d.hpp>
#include <accelerated-arrays/image.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

// #include <iostream>

namespace tracker {
namespace {
class RectifierImplementation : public StereoRectifier {
private:
    Eigen::Matrix4d Q;
    std::array<std::shared_ptr<const Camera>, 2> rectifiedCameras;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RectifierImplementation(
        int width, int height,
        const std::array<api::CameraParameters, 2> &intrinsics,
        const odometry::Parameters &parameters)
    {
        const float focalLengthFactor = parameters.tracker.rectificationZoom;
        const auto &pt = parameters.tracker;
        assert(pt.useStereo);
        const size_t camCount = pt.useStereo ? 2 : 1;

        Q = Eigen::Matrix4d::Identity();

        cv::Matx33d cvCamMatrices[2];
        for (size_t cameraInd = 0; cameraInd < camCount; ++cameraInd) {
            auto &cam = cvCamMatrices[cameraInd];
            auto intrPinhole = intrinsics[cameraInd];
            intrPinhole.focalLengthX *= focalLengthFactor;
            intrPinhole.focalLengthY *= focalLengthFactor;
            cam = cv::Matx33f::zeros();
            cam(0, 0) = intrPinhole.focalLengthX;
            cam(1, 1) = intrPinhole.focalLengthY;
            cam(2, 2) = 1;
            cam(0, 2) = intrPinhole.principalPointX;
            cam(1, 2) = intrPinhole.principalPointY;
        }

        // std::cout << "maxDisparity " << maxDisparity << std::endl;

        // first -> imu -> second = imuToSecond * firstToImu = secondImuToCamera * imuToCamera^{-1}
        Eigen::Matrix4d firstToSecond = parameters.secondImuToCamera * parameters.imuToCamera.inverse();
        cv::Matx33d cvR;
        cv::Matx31d cvT;
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) cvR(i, j) = firstToSecond(i, j);
            cvT(i) = firstToSecond(i, 3);
        }
        // std::cout << "cvR " << cvR << std::endl << "cvT " << cvT << std::endl;

        cv::Matx44d cvQ;
        cv::Matx33d cvR1, cvR2;
        cv::Matx34d cvP1, cvP2;
        std::vector<double> distort = { 0, 0, 0, 0, 0 };
        cv::stereoRectify(cvCamMatrices[0], distort, cvCamMatrices[1], distort, cv::Size(width, height),
            cvR, cvT, cvR1, cvR2, cvP1, cvP2, cvQ);

        for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) Q(i, j) = cvQ(i, j);
        for (size_t cameraInd = 0; cameraInd < 2; ++cameraInd) {
            const auto &cvR = cameraInd == 0 ? cvR1 : cvR2;
            const auto &cvP = cameraInd == 0 ? cvP1 : cvP2;
            Eigen::Matrix3d rectifyRot;
            for (int i=0; i<3; ++i) {
                for (int j=0; j<3; ++j) rectifyRot(i, j) = cvR(i, j);
            }
            // after this, rectifyRot maps: rectified -> unrectified
            rectifyRot = rectifyRot.transpose().eval();

            // NOTE: skipping the fourth column of P on purpose and just extracting the pinhole camera matrix!
            api::CameraParameters rectified;
            rectified.focalLengthX = cvP(0, 0);
            rectified.focalLengthY = cvP(1, 1);
            rectified.principalPointX = cvP(0, 2);
            rectified.principalPointY = cvP(1, 2);
            rectifiedCameras[cameraInd] = Camera::buildPinhole(rectified, {}, width, height, &rectifyRot);

            // OpenCV's Q (disparity-to-depth) mapping gives its results in the rectified camera coordinates,
            // but we need the depth information in the unrectified (original) camera coordinates, because
            // these can be used as-is without further camera mapping
            if (cameraInd == 0) {
                Eigen::Matrix4d Qrot = Eigen::Matrix4d::Identity();
                Qrot.topLeftCorner<3, 3>() = rectifyRot;
                Q = (Qrot * Q).eval();
            }

            // rectifyPInv[cameraInd] = rectifyP[cameraInd].inverse();
            // std::cout << "cvR " << cvR << std::endl;
            // std::cout << "cvP " << cvP << std::endl;
        }

        // std::cout << "Q " << Q << std::endl;
    }

    std::array<std::shared_ptr<const Camera>, 2> getRectifiedCameras() const final {
        return rectifiedCameras;
    }

    Eigen::Matrix4d getDepthQMatrix() const final {
        return Q;
    }
};
}

StereoRectifier::~StereoRectifier() = default;
std::unique_ptr<StereoRectifier> StereoRectifier::build(
    int w, int h,
    const std::array<api::CameraParameters, 2> &intrinsics,
    const odometry::Parameters &parameters)
{
    return std::unique_ptr<StereoRectifier>(new RectifierImplementation(w, h, intrinsics, parameters));
}
}
