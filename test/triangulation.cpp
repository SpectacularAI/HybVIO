// All the numerical comparison tests should pass. If the algorithms are changed then the tests should
// be updated or clearly deprecated.

#include "catch2/catch.hpp"
#include "helpers.hpp"

#include "../src/odometry/parameters.hpp"
#include "../src/odometry/ekf.hpp"
#include "../src/odometry/triangulation.hpp"
#include "../src/views/visualization_internals.hpp"
#include "../src/tracker/camera.hpp"
#include "../src/tracker/util.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace {
Eigen::VectorXd odometryStateToX(odometry::EKF &ekf, int poseCount) {
    Eigen::VectorXd stateVec = ekf.getState();
    constexpr int POSE_SIZE = odometry::POSE_DIM;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(POSE_SIZE * poseCount + 1);
    for (int i = 0; i < poseCount; ++i) {
        const int idxPos = i == 0 ? odometry::POS : (odometry::CAM + ((i-1) * POSE_SIZE));
        const int idxOri = i == 0 ? odometry::ORI : (odometry::CAM + ((i-1) * POSE_SIZE) + 3);
        x.segment<3>(i * POSE_SIZE) = stateVec.segment<3>(idxPos);
        x.segment<4>(i * POSE_SIZE + 3) = stateVec.segment<4>(idxOri).normalized();
    }
    x(x.size() - 1) = 0;
    return x;
};

Eigen::MatrixXd outToDpf(const odometry::TriangulationArgsOut &out) {
    const size_t poseCount = out.dpfdp.size();
    Eigen::MatrixXd dpf = Eigen::MatrixXd::Zero(3, odometry::POSE_DIM * poseCount + 1);
    for (size_t j = 0; j < poseCount; ++j) {
        dpf.block<3, 3>(0, odometry::POSE_DIM * j) = out.dpfdp[j];
        dpf.block<3, 4>(0, odometry::POSE_DIM * j + 3) = out.dpfdq[j];
    }
    dpf.block<3, 1>(0, odometry::POSE_DIM * poseCount) = out.dpfdt;
    return dpf;
}

void sumStereoDerivatives(odometry::TriangulationArgsOut &o) {
    assert(o.dpfdp.size() == o.dpfdq.size());
    const size_t n = o.dpfdp.size() / 2;
    for (size_t i = 0; i < n; ++i) {
        o.dpfdp[i] += o.dpfdp[i + n];
        o.dpfdq[i] += o.dpfdq[i + n];
    }
    o.dpfdp.resize(n);
    o.dpfdq.resize(n);
}
}

TEST_CASE( "visual", "[visual]" ) {
    // Tests numerical accuracy compared to results from our Matlab code.
    //
    // Reminder to myself:
    // The values were generated on the matlab side by
    // datapath = '../data/';
    // captures = list_all_captures(datapath);
    // j = 5;
    // out = run_pivo_newfeatures(fullfile(captures{j}, 'iphone'));
    //
    // if size(obs, 2) == 10 & xf(k) > 51
    //   pause = true;
    //   triangulation_refined_xyz(m,obs,T_imu_cam,t_imu_cam,d_state,11^2*eye(2), pause);
    //   visual_update_refined_xyz(m,xyz,dxyz,y,cameraMatrix,distCoeffs,T_imu_cam,t_imu_cam,d_state, pause);
    //
    // In the triangulation and visual update functions I set t_imu_cam to zero.

    int camPoseCount = 20;
    int stateDim = odometry::INER_DIM + camPoseCount * 7;

    // Construct poses.
    Eigen::Matrix<double, 70, 1> poses; poses << -1.115954259678003, -2.830379937574711, 0.360953864756080, 0.228275363465427, -0.064194730744503, -0.594104812214096, -0.772824444840030, -1.080393253042482, -2.763692958718615, 0.332645073392916, 0.196322489942363, -0.083909476935720, -0.628312037667580, -0.752388564841313, -1.053635192163148, -2.698599740902574, 0.304049959330811, 0.171347617609120, -0.090804163156838, -0.627022749727822, -0.749919482080305, -1.031838101194812, -2.623526076445418, 0.281408008477340, 0.155625729177218, -0.090380891656242, -0.639892913358913, -0.737146980096418, -1.009828260492951, -2.544268915819571, 0.273217018299048, 0.153209864083974, -0.090234014840705, -0.636707261073876, -0.737354342707954, -0.986215006493242, -2.468647298253558, 0.272275808868746, 0.157856184323099, -0.083435652262512, -0.606327170014471, -0.761376924834563, -0.961600705821358, -2.396757542411821, 0.267737813520921, 0.163130732364498, -0.079219306292358, -0.594278868691105, -0.765754228906657, -0.933757923541281, -2.325217937044675, 0.255438002606821, 0.172957779390792, -0.084991869290214, -0.593937386185525, -0.762521999377893, -0.898272888273739, -2.253889975199411, 0.239108878766994, 0.189256086747472, -0.090322497349436, -0.593833321653932, -0.758101862911017, -0.858474881652736, -2.184122374378553, 0.228789583088852, 0.204536006494471, -0.092660683000154, -0.580153035798419, -0.761692686677209;

    Eigen::VectorXd m = Eigen::VectorXd::Zero(stateDim);
    m.segment(odometry::POS, 3) = poses.segment(0, 3);
    m.segment(odometry::ORI, 4) = poses.segment(3, 4);
    for (int i = 0; i < 9; i++) {
        m.segment(odometry::CAM + i * 7, 3) = poses.segment((i + 1) * 7, 3);
        m.segment(odometry::CAM + i * 7 + 3, 4) = poses.segment((i + 1) * 7 + 3, 4);
    }

    vecVector2d imageFeatures;
    vecVector2d featureVelocities;
    odometry::CameraPoseTrail trail;
    odometry::TriangulationArgsIn args {
        .imageFeatures = imageFeatures,
        .featureVelocities = featureVelocities,
        .trail = trail,
        .stereo = false,
        .calculateDerivatives = true,
        .estimateImuCameraTimeShift = true,
        .derivativeTest = true,
        .imuToCameraTimeShift = 0.0,
    };

    odometry::Parameters params;
    tracker::util::automaticCameraParametersWhereUnset(params);
    params.odometry.noiseScale = 1000.0;
    params.odometry.cameraTrailLength = camPoseCount;
    params.odometry.hybridMapSize = 0;
    params.odometry.triangulationConvergenceR = 11.0;
    // params.odometry.useLinearTriangulation = true; // TODO Make a new test case for this?
    auto ekf = odometry::EKF::build(params);
    ekf->setState(m);

    // Construct feature track.
    Eigen::Matrix<double, 10, 2> uv; uv <<
        -0.182574266004879, -0.078574171780591,
        -0.158898685463446, -0.007691759819452,
        -0.131230597106084, -0.013212139610991,
        -0.110637420135181,  0.020800938142075,
        -0.107508132406555,  0.002175057216783,
        -0.108465120810051, -0.080045047328712,
        -0.111911566078740, -0.103534929832195,
        -0.135452929226407, -0.099277664417604,
        -0.165840298753357, -0.093731544303972,
        -0.188661852179662, -0.133908509900881;
    for (int i = 0; i < 10; i++) {
        imageFeatures.push_back(uv.block(i, 0, 1, 2).transpose());
        // Must be non-zero.
        featureVelocities.push_back(Eigen::Vector2d(0.1, 0.1));
    }

    Eigen::Vector3d pf_e(-2.32842, -8.02612, -0.619833);

    int poseCount = imageFeatures.size();
    auto poseTrailIndex = test_helpers::buildRange(poseCount);
    odometry::extractCameraPoseTrail(*ekf, poseTrailIndex, params, false, trail);

    std::vector<bool> mask;
    mask.resize(poseCount, true);

    constexpr unsigned POSE_SIZE = 7;
    auto xToTrail = [&](const Eigen::VectorXd& x, odometry::CameraPoseTrail &trail) {
        Eigen::VectorXd stateVec = ekf->getState(), stateOrig;
        stateOrig = stateVec;
        assert(x.size() == POSE_SIZE * poseCount + 1);
        for (int i = 0; i < poseCount; ++i) {
            const int idxPos = i == 0 ? odometry::POS : (odometry::CAM + ((i-1) * POSE_SIZE));
            const int idxOri = i == 0 ? odometry::ORI : (odometry::CAM + ((i-1) * POSE_SIZE) + 3);
            stateVec.segment<3>(idxPos) = x.segment<3>(i * POSE_SIZE);
            stateVec.segment<4>(idxOri) = x.segment<4>(i * POSE_SIZE + 3).normalized();
        }
        ekf->setState(stateVec);
        odometry::extractCameraPoseTrail(*ekf,
            poseTrailIndex,
            params,
            false,
            trail);
        ekf->setState(stateOrig);
    };

    odometry::Triangulator triangulator(params.odometry);
    params.odometry.cameraTrailLength = poseCount - 1;
    params.odometry.hybridMapSize = 0;

    SECTION( "triangulate" ) {
        odometry::TriangulationArgsOut out;
        odometry::TriangulatorStatus status = triangulator.triangulate(args, out);
        REQUIRE( status == odometry::TriangulatorStatus::OK );
        if (!params.odometry.useLinearTriangulation) {
            REQUIRE( (out.pf - pf_e).array().abs().sum() < 1e-5 );
        }

        auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            return out.pf;
        };

        auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            Eigen::MatrixXd dpf = outToDpf(out);
            assert(dpf.rows() == 3 && dpf.cols() == x.size());
            return dpf;
        };

        Eigen::MatrixXd D = test_helpers::der_check(odometryStateToX(*ekf, poseCount), numeric, analytic);
        REQUIRE( D.cwiseAbs().maxCoeff() < 1e-3 );
    }

    SECTION( "prepareVisualUpdateCheckJacobian" ) {
        odometry::TriangulationArgsOut out;
        odometry::PrepareVisualUpdateArgsIn prepareArgs {
            .triangulationOut = out,
            .featureVelocities = featureVelocities,
            .trail = trail,
            .poseTrailIndex = poseTrailIndex,
            .stateDim = ekf->getStateDim(),
            .useStereo = false,
            .truncated = false,
            .mapPointOffset = -1,
            .estimateImuCameraTimeShift = true,
            .derivativeTest = true,
            .imuToCameraTimeShift = 0.0,
        };

        auto numeric = [&](const Eigen::VectorXd& x) {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            prepareArgs.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            Eigen::MatrixXd H;
            Eigen::VectorXd f;
            odometry::prepareVisualUpdate(prepareArgs, H, f);
            return f;
        };

        auto analytic = [&](const Eigen::VectorXd& x) {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            prepareArgs.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            Eigen::MatrixXd H;
            Eigen::VectorXd f;
            odometry::prepareVisualUpdate(prepareArgs, H, f);
            Eigen::MatrixXd h = Eigen::MatrixXd::Zero(20, 71);

            // Remove derivatives of unrelated variables.
            int j = 0;
            for (int i = 0; i < 3; ++i) {
                h.col(j++) = H.col(odometry::POS + i);
            }
            for (int i = 0; i < 4; ++i) {
                h.col(j++) = H.col(odometry::ORI + i);
            }
            for (int i = 0; i < 7 * (poseCount - 1); ++i) {
                h.col(j++) = H.col(odometry::CAM + i);
            }
            h.col(j++) = H.col(odometry::SFT);
            assert(j == h.cols());
            return h;
        };

        Eigen::MatrixXd D = test_helpers::der_check(odometryStateToX(*ekf, poseCount), numeric, analytic);
        REQUIRE( D.cwiseAbs().maxCoeff() < 1e-6);
    }
}

TEST_CASE( "stereo_visual", "[stereo_visual]" ) {
    int camPoseCount = 10;
    int stateDim = odometry::INER_DIM + camPoseCount * 7;

    // Construct poses.
    Eigen::Matrix<double, 70, 1> poses;
    poses << -0.367827, -15.0661, 0.0399335, -0.745415, 0.487042, 0.328822, 0.314678,
            -0.249844, -15.0911, 0.0486579, -0.738804, 0.484781, 0.335281, 0.332631,
            -0.127223, -15.115, 0.0603704, -0.731083, 0.479407, 0.340445, 0.351686,
            -0.00431178, -15.1348, 0.0773177, -0.724992, 0.473098, 0.342716, 0.370199,
            0.102376, -15.148, 0.0948278, -0.718232, 0.466938, 0.347867, 0.386091,
            0.226335, -15.1556, 0.115942, -0.712628, 0.460881, 0.353862, 0.398023,
            0.350032, -15.1552, 0.136109, -0.707418, 0.456131, 0.359319, 0.407636,
            0.45797, -15.1496, 0.151658, -0.701875, 0.454955, 0.363992, 0.414029,
            0.585777, -15.1363, 0.16722, -0.696256, 0.451043, 0.371664, 0.420804,
            0.71479, -15.1179, 0.179145, -0.69071, 0.448982, 0.376893, 0.427637;

    Eigen::VectorXd m = Eigen::VectorXd::Zero(stateDim);
    m.segment(odometry::POS, 3) = poses.segment(0, 3);
    m.segment(odometry::ORI, 4) = poses.segment(3, 4);
    for (int i = 0; i < 9; i++) {
        m.segment(odometry::CAM + i * 7, 3) = poses.segment((i + 1) * 7, 3);
        m.segment(odometry::CAM + i * 7 + 3, 4) = poses.segment((i + 1) * 7 + 3, 4);
    }

    vecVector2d imageFeatures;
    vecVector2d featureVelocities;
    odometry::CameraPoseTrail trail;
    odometry::TriangulationArgsIn args {
        .imageFeatures = imageFeatures,
        .featureVelocities = featureVelocities,
        .trail = trail,
        .stereo = true,
        .calculateDerivatives = true,
        .estimateImuCameraTimeShift = true,
        .derivativeTest = true,
        .imuToCameraTimeShift = 0.0,
    };

    odometry::Parameters params;
    params.odometry.imuToCameraMatrix = {0, -1, 0, -1, 0, 0, 0, 0, -1};
    params.odometry.secondImuToCameraMatrix = {
            4.92411476e-04,-9.99955101e-01,9.46330107e-03,
            -9.99990741e-01,-4.51929559e-04,4.27944220e-03,
            -4.27497331e-03,-9.46532070e-03,-9.99946065e-01
    };
    tracker::util::automaticCameraParametersWhereUnset(params);
    params.odometry.noiseScale = 1000.0;
    params.odometry.cameraTrailLength = camPoseCount;
    params.odometry.hybridMapSize = 0;
    params.odometry.triangulationConvergenceR = 11.0;
    auto ekf = odometry::EKF::build(params);
    ekf->setState(m);

    // Construct feature track.
    Eigen::Matrix<double, 10, 2> uv; uv <<
            -0.124468, -0.177301,
            -0.120764, -0.202625,
            -0.129309, -0.223704,
            -0.141031, -0.247501,
            -0.146423, -0.274626,
            -0.150899, -0.295449,
            -0.152864, -0.31037,
            -0.147448, -0.319061,
            -0.146217, -0.33278,
            -0.143194, -0.342048;
    Eigen::Matrix<double, 10, 2> secondUv; secondUv <<
            -0.126105, -0.18367,
            -0.121975, -0.2084,
            -0.131587, -0.230039,
            -0.142692, -0.25362,
            -0.147776, -0.280209,
            -0.152159, -0.300713,
            -0.153695, -0.315995,
            -0.14808, -0.324273,
            -0.147324, -0.33733,
            -0.143747, -0.345866;
    // make sure there's enough baseline so all the above features can be
    // triangulated successfully from one "image pair"
    secondUv *= 1.1;

    for (int i = 0; i < 10; i++) {
        imageFeatures.push_back(uv.block(i, 0, 1, 2).transpose());
        featureVelocities.push_back(Eigen::Vector2d(0.1, 0.1));
    }
    for (int i = 0; i < 10; i++) {
        imageFeatures.push_back(secondUv.block(i, 0, 1, 2).transpose());
        featureVelocities.push_back(Eigen::Vector2d(0.1, 0.1));
    }

    int poseCount = imageFeatures.size() / 2;
    auto poseTrailIndex = test_helpers::buildRange(poseCount);
    extractCameraPoseTrail(*ekf, poseTrailIndex, params, false, trail);

    std::vector<bool> mask;
    mask.resize(poseCount, true);
    constexpr unsigned POSE_SIZE = 7;

    const Eigen::Matrix4d transformSecondToFirstCamera = params.imuToCamera * params.secondImuToCamera.inverse();
    for (auto i : poseTrailIndex) {
        Eigen::Vector2d ip1 = uv.row(i).transpose();
        Eigen::Vector2d ip0 = secondUv.row(i).transpose();
        auto &camPose = trail.at(i);
        camPose.hasFeature3D = true;
        REQUIRE(odometry::triangulateStereoFeatureIdp(
            ip0,
            ip1,
            transformSecondToFirstCamera,
            camPose.feature3DIdp,
            &camPose.feature3DCov));
    }

    const odometry::CameraPoseTrail trail0 = trail;

    auto xToTrail = [&](const Eigen::VectorXd& x, odometry::CameraPoseTrail &trail) {
        Eigen::VectorXd stateVec = ekf->getState(), stateOrig;
        stateOrig = stateVec;
        assert(x.size() == POSE_SIZE * poseCount + 1);
        for (int i = 0; i < poseCount; ++i) {
            const int idxPos = i == 0 ? odometry::POS : (odometry::CAM + ((i-1) * POSE_SIZE));
            const int idxOri = i == 0 ? odometry::ORI : (odometry::CAM + ((i-1) * POSE_SIZE) + 3);
            stateVec.segment<3>(idxPos) = x.segment<3>(i * POSE_SIZE);
            stateVec.segment<4>(idxOri) = x.segment<4>(i * POSE_SIZE + 3);
        }
        ekf->setState(stateVec);
        odometry::extractCameraPoseTrail(*ekf,
            poseTrailIndex,
            params,
            true,
            trail);
        ekf->setState(stateOrig);
        for (auto i : poseTrailIndex) {
            auto &camPose = trail.at(i);
            const auto &camPose0 = trail0.at(i);
            camPose.hasFeature3D = true;
            camPose.feature3DIdp = camPose0.feature3DIdp;
            camPose.feature3DCov = camPose0.feature3DCov;
        }
    };

    params.odometry.cameraTrailLength = poseCount - 1;
    params.odometry.hybridMapSize = 0;
    odometry::Triangulator triangulator(params.odometry);

    SECTION( "triangulate" ) {
        odometry::TriangulationArgsOut out;
        auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            return out.pf;
        };

        auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
            xToTrail(x, trail);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            sumStereoDerivatives(out);
            Eigen::MatrixXd dpf = outToDpf(out);
            assert(dpf.rows() == 3 && dpf.cols() == x.size());
            return dpf;
        };

        Eigen::MatrixXd D = test_helpers::der_check(odometryStateToX(*ekf, poseCount), numeric, analytic);
        REQUIRE( D.cwiseAbs().maxCoeff() < 1e-4 );
    }

    SECTION( "prepareVisualUpdateCheckJacobian" ) {
        odometry::TriangulationArgsOut out;
        odometry::PrepareVisualUpdateArgsIn prepareArgs {
            .triangulationOut = out,
            .featureVelocities = featureVelocities,
            .trail = trail,
            .poseTrailIndex = poseTrailIndex,
            .stateDim = ekf->getStateDim(),
            .useStereo = true,
            .truncated = false,
            .mapPointOffset = -1,
            .estimateImuCameraTimeShift = true,
            .derivativeTest = true,
            .imuToCameraTimeShift = 0.0,
        };

        auto numeric = [&](const Eigen::VectorXd& x) {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            prepareArgs.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            sumStereoDerivatives(out);

            Eigen::MatrixXd H;
            Eigen::VectorXd f;
            odometry::prepareVisualUpdate(prepareArgs, H, f);
            return f;
        };

        auto analytic = [&](const Eigen::VectorXd& x) {
            xToTrail(x, trail);
            args.imuToCameraTimeShift = x(x.size() - 1);
            prepareArgs.imuToCameraTimeShift = x(x.size() - 1);
            assert(triangulator.triangulate(args, out) == odometry::TriangulatorStatus::OK);
            sumStereoDerivatives(out);

            Eigen::MatrixXd H;
            Eigen::VectorXd f;
            odometry::prepareVisualUpdate(prepareArgs, H, f);
            Eigen::MatrixXd h = Eigen::MatrixXd::Zero(40, 71);

            // Remove derivatives of unrelated variables.
            int j = 0;
            for (int i = 0; i < 3; ++i) {
                h.col(j++) = H.col(odometry::POS + i);
            }
            for (int i = 0; i < 4; ++i) {
                h.col(j++) = H.col(odometry::ORI + i);
            }
            for (int i = 0; i < 7 * (poseCount - 1); ++i) {
                h.col(j++) = H.col(odometry::CAM + i);
            }
            h.col(j++) = H.col(odometry::SFT);
            assert(j == h.cols());

            return h;
        };

        Eigen::MatrixXd D = test_helpers::der_check(odometryStateToX(*ekf, poseCount), numeric, analytic);
        REQUIRE(D.cwiseAbs().maxCoeff() < 1e-5);
    }
}

TEST_CASE( "pinv", "[pinv]" ) {
    Eigen::Matrix<double, 2, 3> m; m << 1, 3, -1, 4, 2, -3;
    // Matlab pinv() function result.
    Eigen::Matrix<double, 3, 2> pinvm; pinvm <<
        -0.153333,  0.206667,
        0.406667, -0.113333,
        0.0666667, -0.133333;
    REQUIRE( (odometry::pinv(m.transpose()) - pinvm.transpose()).array().abs().sum() < 1e-5 );
}

TEST_CASE( "triangulateWithTwoCameras", "[triangulateWithTwoCameras]" ) {
    // the camera rays form a right triangle
    //
    //      o  intersection = (1, 2, 1)
    //     /|
    //    / |
    //   /  |    ^ z
    //  o---o    |
    // p0   p1   *--> y
    //
    const Eigen::Vector2d ip0(0, 1);
    const odometry::CameraPose pose0 = {
        .p = Eigen::Vector3d(1, 1, 0),
        .R = Eigen::Matrix3d::Identity()
    };
    const Eigen::Vector2d ip1(0, 0);
    const odometry::CameraPose pose1 = {
        .p = Eigen::Vector3d(1, 2, 0),
        .R = Eigen::Matrix3d::Identity()
    };

    const odometry::TwoCameraTriangulationArgsIn args {
        .pose0 = pose0,
        .pose1 = pose1,
        .ip0 = ip0,
        .ip1 = ip1,
        .calculateDerivatives = false,
        .estimateImuCameraTimeShift = false,
    };
    Eigen::Vector3d pf = odometry::triangulateWithTwoCameras(args);
    Eigen::Vector3d pf_e(0, 1, 1);
    REQUIRE( (pf - pf_e).array().abs().sum() < 1e-5 );
}

TEST_CASE( "der_triangulateWithTwoCameras", "[der_triangulateWithTwoCameras]" ) {
    using odometry::CameraPose;
    const Eigen::Vector2d ip0(0.349580805, 0.567773197);
    const Eigen::Vector2d ip1(0.212656222, 0.436937714);
    // Must be non-zero.
    const Eigen::Vector2d v0(0.1, 0.1);
    const Eigen::Vector2d v1(0.1, 0.1);
    constexpr int cols = 2 * 7 + 1;

    CameraPose pose0, pose1;
    odometry::TwoCameraTriangulationArgsIn args {
        .pose0 = pose0,
        .pose1 = pose1,
        .ip0 = ip0,
        .ip1 = ip1,
        .velocity0 = &v0,
        .velocity1 = &v1,
        .calculateDerivatives = true,
        .estimateImuCameraTimeShift = true,
        .derivativeTest = true,
        .imuToCameraTimeShift = 0.0,
    };

    auto vectorToInputs = [&](const Eigen::VectorXd& x, CameraPose &pose0, CameraPose &pose1) {
        const Eigen::Vector3d p0 = x.segment(0, 3);
        const Eigen::Vector4d q0 = x.segment(3, 4);
        const Eigen::Vector3d p1 = x.segment(7, 3);
        const Eigen::Vector4d q1 = x.segment(10, 4);

        pose0.p = p0;
        pose0.R = odometry::util::quat2rmat_d(q0, pose0.dR);
        pose1.p = p1;
        pose1.R = odometry::util::quat2rmat_d(q1, pose1.dR);
    };

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        vectorToInputs(x, pose0, pose1);
        args.imuToCameraTimeShift = x(x.size() - 1);
        Eigen::Matrix<double, 3, cols> dpf;
        return odometry::triangulateWithTwoCameras(args, &dpf);
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        vectorToInputs(x, pose0, pose1);
        args.imuToCameraTimeShift = x(x.size() - 1);
        Eigen::Matrix<double, 3, cols> dpf;
        odometry::triangulateWithTwoCameras(args, &dpf);
        return dpf;
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cols);
    x0.segment(0, 3) = Eigen::Vector3d(-4.18946568, -2.60358733,  1.30011701); // p0
    x0.segment(3, 4) = Eigen::Vector4d(0.41490869, -1.20760051, -0.41931339, -0.86055931).normalized();
    x0.segment(7, 3) = Eigen::Vector3d(-3.87749592, -2.79185809, 0.962672003); // p1
    x0.segment(10, 4) =  Eigen::Vector4d(2.3420597 ,  0.32444083,  0.21743605, -0.1400614).normalized();
    x0(cols - 1) = 0.0;
    Eigen::MatrixXd D = test_helpers::der_check(x0, numeric, analytic);

    REQUIRE( D.norm() < 2e-6 );
}

TEST_CASE( "der_inverseDepth", "[der_inverseDepth]" ) {
    using odometry::CameraPose;
    const Eigen::Vector3d p0(0.349580805, 0.567773197, 1.436937714);

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Matrix3d dip;
        return odometry::inverseDepth(x, dip);
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::Matrix3d dip;
        odometry::inverseDepth(x, dip);
        return dip;
    };

    Eigen::MatrixXd D = test_helpers::der_check(p0, numeric, analytic);

    //std::cout << D << std::endl;
    REQUIRE( D.norm() < 1e-5 );
}

TEST_CASE( "inverseDepthSecondDerivative", "[inverseDepthSecondDerivative]" ) {
    using odometry::CameraPose;
    const Eigen::Vector3d p0(0.349580805, 0.567773197, 1.436937714);

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::Matrix3d dip;
        Eigen::Matrix<double, 9, 1> flat;
        odometry::inverseDepth(x, dip);
        for (int i = 0; i < 3; ++i) flat.segment<3>(i * 3) = dip.col(i);
        return flat;
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::Matrix3d dip, ddip[3];
        odometry::inverseDepth(x, dip, ddip);
        Eigen::Matrix<double, 9, 3> flat;
        for (int i = 0; i < 3; ++i) flat.block<3, 3>(i * 3, 0) = ddip[i];
        return flat;
    };

    Eigen::MatrixXd D = test_helpers::der_check(p0, numeric, analytic);

    //std::cout << D << std::endl;
    REQUIRE( D.norm() < 1e-5 );
}
