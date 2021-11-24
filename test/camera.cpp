#include "catch2/catch.hpp"
#include "helpers.hpp"

#include "../src/tracker/camera.hpp"
#include <cmath>

TEST_CASE( "fisheye camera", "[camera]" ) {
    for (bool distorted : { false, true }) {
        api::CameraParameters intrinsic;
        intrinsic.focalLengthX = 10;
        intrinsic.focalLengthY = 11;
        intrinsic.principalPointX = 5;
        intrinsic.principalPointY = 5.5;
        std::unique_ptr<const tracker::Camera> camera;
        if (distorted) {
            // actual values from a RealSense camera
            std::vector<double> coeff = {
                -0.00200599804520607,
                0.03895416110754013,
                -0.03715667128562927,
                0.0061612860299646854
            };
            camera = tracker::Camera::buildFisheye(intrinsic, coeff);
        }
        else {
            camera = tracker::Camera::buildFisheye(intrinsic);
        }

        Eigen::Vector3d principalAxis;
        camera->pixelToRay({ 5, 5.5 }, principalAxis);
        REQUIRE( principalAxis.segment<2>(0).norm() < 1e-6 );
        REQUIRE( std::abs(principalAxis.z() - 1) < 1e-6 );

        const Eigen::Vector3d v(1,-2,3);
        Eigen::Vector2d p;
        REQUIRE( camera->rayToPixel(v, p) );

        REQUIRE( camera->isValidPixel(p) );
        REQUIRE( !camera->isValidPixel(1000, 10000) );

        Eigen::Vector3d proj;
        camera->pixelToRay(p, proj);
        REQUIRE( std::abs(v.normalized().z() - proj.z()) < 1e-6 );
        REQUIRE(p.x() > intrinsic.principalPointX);
        REQUIRE(p.y() < intrinsic.principalPointY);

        // serialization test
        auto serialized = camera->serialize();
        auto camera2 = tracker::Camera::deserialize(serialized);

        const Eigen::Vector2d testPix(5.23, 3.12);
        Eigen::Vector3d rayBefore, rayAfter;
        REQUIRE( camera->pixelToRay(testPix, rayBefore) );
        REQUIRE( camera2->pixelToRay(testPix, rayAfter) );
        // std::cout << serialized << std::endl;
        // std::cout << "before " << rayBefore << std::endl;
        // std::cout << "after " << rayAfter << std::endl;
        REQUIRE( (rayBefore - rayAfter).array().abs().sum() < 1e-4 );


        // check derivative
        auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::Vector2d p;
            camera->rayToPixel(x, p);
            return p;
        };

        auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
            Eigen::Vector2d p;
            tracker::Camera::Matrix2x3 jacob;
            camera->rayToPixel(x, p, &jacob);
            // std::cout << jacob << std::endl;
            return jacob;
        };

        Eigen::VectorXd x0 = v;
        Eigen::MatrixXd D = test_helpers::der_check(x0, numeric, analytic);
        //std::cout << D << std::endl;
        REQUIRE( D.norm() < 1e-4 );
    }
}

TEST_CASE( "pinhole/projectToCamera", "[camera]") {
    api::CameraParameters intrinsic;
    intrinsic.focalLengthX = 1000;
    intrinsic.focalLengthY = 1000;
    intrinsic.principalPointX = 360;
    intrinsic.principalPointY = 640;
    auto camera = tracker::Camera::buildPinhole(intrinsic);
    Eigen::Vector3d p(-0.25, 0.11, 2);
    // Result given by our matlab function: [ipe, dipe] = undistort(p, C, zeros(1, 5));
    Eigen::Vector2d ipe(235, 695);
    Eigen::Matrix<double, 3, 2> dipe; dipe <<
        500,  0,
        0,    500,
        62.5, -27.5;

    Eigen::Matrix<double, 2, 3> dip;
    Eigen::Vector2d ip;
    REQUIRE( camera->rayToPixel(p, ip, &dip) );
    REQUIRE( (ip - ipe).array().abs().sum() < 1e-5 );
    REQUIRE( (dip.transpose() - dipe).array().abs().sum() < 1e-5 );

    // serialization test
    auto serialized = camera->serialize();
    auto camera2 = tracker::Camera::deserialize(serialized);

    ip.setZero();
    dip.setZero();
    REQUIRE( camera2->rayToPixel(p, ip, &dip) );
    REQUIRE( (ip - ipe).array().abs().sum() < 1e-5 );
    REQUIRE( (dip.transpose() - dipe).array().abs().sum() < 1e-5 );

}

TEST_CASE("pinhole/distortion/projectToCamera", "[camera]") {
    api::CameraParameters intrinsic;
    intrinsic.focalLengthX = 1.31841527e+03;
    intrinsic.focalLengthY = 1.31745365e+03;
    intrinsic.principalPointX = 9.49043714e+02;
    intrinsic.principalPointY = 5.31894317e+02;
    auto camera = tracker::Camera::buildPinhole(
            intrinsic,
            {0.20740335, -0.28361953, -0.10090323},
            2000, 3000);

    Eigen::Vector3d ray0 = {0.26726124, 0.53452248, 0.80178373};
    Eigen::Vector2d pixel0 = {1393.07961912, 1419.31839027};

    Eigen::Vector2d pixelp;
    camera->rayToPixel(ray0, pixelp, nullptr);
    Eigen::Vector3d rayp;
    camera->pixelToRay(pixel0, rayp);

    REQUIRE( camera->isValidPixel(pixel0) );
    REQUIRE( camera->isValidPixel(0, 0) );
    REQUIRE( camera->isValidPixel(1999, 2999) );
    REQUIRE( !camera->isValidPixel(2000, 0) );
    REQUIRE( !camera->isValidPixel(1999, -1) );

    REQUIRE( (pixelp - pixel0).array().abs().sum() < 1e-3 );
    REQUIRE( (rayp - ray0).array().abs().sum() < 1e-4 );

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Vector2d p;
        camera->rayToPixel(x, p);
        return p;
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::Vector2d p;
        tracker::Camera::Matrix2x3 jacob;
        camera->rayToPixel(x, p, &jacob);
        return jacob;
    };

    Eigen::MatrixXd D = test_helpers::der_check(ray0, numeric, analytic);
    // std::cout << D << std::endl;
    REQUIRE(D.norm() < 1e-3);

    // serialization test
    auto serialized = camera->serialize();
    auto camera2 = tracker::Camera::deserialize(serialized);

    rayp.setZero();
    REQUIRE( camera2->pixelToRay(pixel0, rayp) );
    REQUIRE( (rayp - ray0).array().abs().sum() < 1e-4 );
}

TEST_CASE( "pinhole/der_projectToCamera", "[camera]" ) {
    api::CameraParameters intrinsic;
    intrinsic.focalLengthX = 1000;
    intrinsic.focalLengthY = 1000;
    intrinsic.principalPointX = 360;
    intrinsic.principalPointY = 640;
    auto camera = tracker::Camera::buildPinhole(intrinsic);

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Vector2d ip;
        REQUIRE( camera->rayToPixel(x, ip) );
        return ip;
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::Matrix<double, 2, 3> dip;
        Eigen::Vector2d ip;
        REQUIRE( camera->rayToPixel(x, ip, &dip) );
        return dip;
    };

    Eigen::Vector3d p(-0.25, 0.11, 2);
    Eigen::MatrixXd D = test_helpers::der_check(p, numeric, analytic);
    //std::cout << D << std::endl;
    REQUIRE( D.cwiseAbs().maxCoeff() < 1e-3 );
}
