// All the numerical comparison tests should pass. If the algorithms are changed then the tests should
// be updated or clearly deprecated.

#include "catch2/catch.hpp"
#include "helpers.hpp"

#include "../src/odometry/parameters.hpp"
#include "../src/odometry/ekf.hpp"
#include "../src/odometry/triangulation.hpp"
#include "../src/odometry/util.hpp"
#include "../src/views/visualization_internals.hpp"
#include "../src/tracker/camera.hpp"
#include "../src/tracker/util.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// Verify a formula used for computing chi-squared test value in EKF::updateVisualTrack.
TEST_CASE( "chi-squared innovation test", "[EKF]" ) {
    Eigen::Matrix<double, 20, 20> M;
    M.block(0, 0, 20, 10) <<
        0.5742,    0.0892,    0.4306,    0.1112,    0.4154,    0.1286,    0.4022,    0.1333,    0.3931,    0.1336,
        0.0892,    1.5660,   -0.0077,    1.3647,   -0.0613,    1.3016,   -0.0806,    1.2433,   -0.0771,    1.1955,
        0.4306,   -0.0077,    0.5371,    0.0184,    0.4057,    0.0390,    0.3947,    0.0471,    0.3860,    0.0503,
        0.1112,    1.3647,    0.0184,    1.4136,   -0.0334,    1.2360,   -0.0525,    1.1829,   -0.0501,    1.1390,
        0.4154,   -0.0613,    0.4057,   -0.0334,    0.5193,   -0.0113,    0.3888,   -0.0015,    0.3805,    0.0032,
        0.1286,    1.3016,    0.0390,    1.2360,   -0.0113,    1.3059,   -0.0304,    1.1361,   -0.0288,    1.0955,
        0.4022,   -0.0806,    0.3947,   -0.0525,    0.3888,   -0.0304,    0.5014,   -0.0204,    0.3726,   -0.0153,
        0.1333,    1.2433,    0.0471,    1.1829,   -0.0015,    1.1361,   -0.0204,    1.2121,   -0.0192,    1.0534,
        0.3931,   -0.0771,    0.3860,   -0.0501,    0.3805,   -0.0288,    0.3726,   -0.0192,    0.4863,   -0.0144,
        0.1336,    1.1955,    0.0503,    1.1390,    0.0032,    1.0955,   -0.0153,    1.0534,   -0.0144,    1.1392,
        0.3875,   -0.0556,    0.3793,   -0.0303,    0.3734,   -0.0104,    0.3656,   -0.0019,    0.3586,    0.0021,
        0.1324,    1.1608,    0.0513,    1.1072,    0.0052,    1.0661,   -0.0130,    1.0263,   -0.0123,    0.9929,
        0.3837,   -0.0320,    0.3743,   -0.0085,    0.3677,    0.0100,    0.3598,    0.0174,    0.3531,    0.0204,
        0.1289,    1.1298,    0.0497,    1.0787,    0.0046,    1.0397,   -0.0133,    1.0018,   -0.0128,    0.9700,
        0.3797,   -0.0127,    0.3692,    0.0093,    0.3621,    0.0265,    0.3541,    0.0328,    0.3477,    0.0351,
        0.1251,    1.0946,    0.0482,    1.0460,    0.0042,    1.0092,   -0.0133,    0.9733,   -0.0130,    0.9433,
        0.3784,    0.0196,    0.3658,    0.0395,    0.3577,    0.0550,    0.3494,    0.0599,    0.3432,    0.0610,
        0.1172,    1.0612,    0.0426,    1.0148,   -0.0002,    0.9799,   -0.0173,    0.9459,   -0.0171,    0.9175,
        0.3798,    0.0545,    0.3649,    0.0722,    0.3555,    0.0861,    0.3468,    0.0894,    0.3408,    0.0892,
        0.1124,    1.0387,    0.0392,    0.9941,   -0.0029,    0.9607,   -0.0197,    0.9281,   -0.0197,    0.9008;
    M.block(0, 10, 20, 10) <<
        0.3875,    0.1324,    0.3837,    0.1289,    0.3797,    0.1251,    0.3784,    0.1172,    0.3798,    0.1124,
       -0.0556,    1.1608,   -0.0320,    1.1298,   -0.0127,    1.0946,    0.0196,    1.0612,    0.0545,    1.0387,
        0.3793,    0.0513,    0.3743,    0.0497,    0.3692,    0.0482,    0.3658,    0.0426,    0.3649,    0.0392,
       -0.0303,    1.1072,   -0.0085,    1.0787,    0.0093,    1.0460,    0.0395,    1.0148,    0.0722,    0.9941,
        0.3734,    0.0052,    0.3677,    0.0046,    0.3621,    0.0042,    0.3577,   -0.0002,    0.3555,   -0.0029,
       -0.0104,    1.0661,    0.0100,    1.0397,    0.0265,    1.0092,    0.0550,    0.9799,    0.0861,    0.9607,
        0.3656,   -0.0130,    0.3598,   -0.0133,    0.3541,   -0.0133,    0.3494,   -0.0173,    0.3468,   -0.0197,
       -0.0019,    1.0263,    0.0174,    1.0018,    0.0328,    0.9733,    0.0599,    0.9459,    0.0894,    0.9281,
        0.3586,   -0.0123,    0.3531,   -0.0128,    0.3477,   -0.0130,    0.3432,   -0.0171,    0.3408,   -0.0197,
        0.0021,    0.9929,    0.0204,    0.9700,    0.0351,    0.9433,    0.0610,    0.9175,    0.0892,    0.9008,
        0.4737,    0.0035,    0.3479,    0.0025,    0.3429,    0.0017,    0.3390,   -0.0029,    0.3372,   -0.0058,
        0.0035,    1.0901,    0.0212,    0.9476,    0.0353,    0.9222,    0.0603,    0.8976,    0.0877,    0.8820,
        0.3479,    0.0212,    0.4647,    0.0196,    0.3393,    0.0182,    0.3360,    0.0130,    0.3349,    0.0096,
        0.0025,    0.9476,    0.0196,    1.0484,    0.0332,    0.9034,    0.0575,    0.8801,    0.0840,    0.8654,
        0.3429,    0.0353,    0.3393,    0.0332,    0.4565,    0.0312,    0.3329,    0.0255,    0.3323,    0.0218,
        0.0017,    0.9222,    0.0182,    0.9034,    0.0312,    1.0019,    0.0546,    0.8589,    0.0802,    0.8453,
        0.3390,    0.0603,    0.3360,    0.0575,    0.3329,    0.0546,    0.4523,    0.0481,    0.3317,    0.0438,
       -0.0029,    0.8976,    0.0130,    0.8801,    0.0255,    0.8589,    0.0481,    0.9595,    0.0728,    0.8260,
        0.3372,    0.0877,    0.3349,    0.0840,    0.3323,    0.0802,    0.3317,    0.0728,    0.4542,    0.0679,
       -0.0058,    0.8820,    0.0096,    0.8654,    0.0218,    0.8453,    0.0438,    0.8260,    0.0679,    0.9354;
    M *= 1e3;

    Eigen::Matrix<double, 20, 1> v;
    v << 0.1467,   -1.0488,    3.0265,    0.2151,   -3.0635,   -0.3286,   -0.3737,   -4.6158,   -0.9681,    5.9890,    -0.5314,    6.0519,   -0.4472,    0.5639,    1.5391,   -3.5595,    2.6163,   -7.4469,   -2.2255,    3.9917;

    // Matlab expression: t = v'/M*v;
    double t = M.ldlt().solve(v).transpose() * v;
    REQUIRE( std::abs(t - 1.7626) < 1e-1 );
}

TEST_CASE( "der_predict", "[EKF]" ) {
    int camPoseCount = 20;
    int stateDim = odometry::INER_DIM + camPoseCount * 7;
    Eigen::Matrix<double, 70, 1> poses; poses << -1.115954259678003, -2.830379937574711, 0.360953864756080, 0.228275363465427, -0.064194730744503, -0.594104812214096, -0.772824444840030, -1.080393253042482, -2.763692958718615, 0.332645073392916, 0.196322489942363, -0.083909476935720, -0.628312037667580, -0.752388564841313, -1.053635192163148, -2.698599740902574, 0.304049959330811, 0.171347617609120, -0.090804163156838, -0.627022749727822, -0.749919482080305, -1.031838101194812, -2.623526076445418, 0.281408008477340, 0.155625729177218, -0.090380891656242, -0.639892913358913, -0.737146980096418, -1.009828260492951, -2.544268915819571, 0.273217018299048, 0.153209864083974, -0.090234014840705, -0.636707261073876, -0.737354342707954, -0.986215006493242, -2.468647298253558, 0.272275808868746, 0.157856184323099, -0.083435652262512, -0.606327170014471, -0.761376924834563, -0.961600705821358, -2.396757542411821, 0.267737813520921, 0.163130732364498, -0.079219306292358, -0.594278868691105, -0.765754228906657, -0.933757923541281, -2.325217937044675, 0.255438002606821, 0.172957779390792, -0.084991869290214, -0.593937386185525, -0.762521999377893, -0.898272888273739, -2.253889975199411, 0.239108878766994, 0.189256086747472, -0.090322497349436, -0.593833321653932, -0.758101862911017, -0.858474881652736, -2.184122374378553, 0.228789583088852, 0.204536006494471, -0.092660683000154, -0.580153035798419, -0.761692686677209;

    Eigen::VectorXd m = Eigen::VectorXd::Zero(stateDim);
    m.segment(odometry::POS, 3) = poses.segment(0, 3);
    m.segment(odometry::ORI, 4) = poses.segment(3, 4);
    for (int i = 0; i < 9; i++) {
        m.segment(odometry::CAM + i * 7, 3) = poses.segment((i + 1) * 7, 3);
        m.segment(odometry::CAM + i * 7 + 3, 4) = poses.segment((i + 1) * 7 + 3, 4);
    }

    const double t = 0.01;
    const double dt = 0.01;
    Eigen::Vector3d gyro; gyro << 0.188914, -0.313109, -0.032521;
    Eigen::Vector3d acc; acc << 0.182453, 7.46259, 2.25091;

    odometry::Parameters params;
    params.odometry.cameraTrailLength = 5;
    params.odometry.hybridMapSize = 0;
    auto odometry0 = odometry::EKF::build(params);
    odometry0->setFirstSampleTime(t);

    auto numeric = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        auto odometry = odometry0->clone();
        Eigen::VectorXd m = odometry->getState();
        m.segment<odometry::INER_DIM>(0) = x;
        odometry->setState(m);
        odometry->predict(t + dt, gyro, acc);
        return odometry->getState().segment(0, odometry::INER_DIM);
    };

    auto analytic = [&](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        auto odometry = odometry0->clone();
        Eigen::VectorXd m = odometry->getState();
        m.segment<odometry::INER_DIM>(0) = x;
        odometry->setState(m);
        odometry->predict(t + dt, gyro, acc);
        return odometry->getDydx().block(0, 0, odometry::INER_DIM, odometry::INER_DIM);
    };

    Eigen::MatrixXd D = test_helpers::der_check(m.block(0, 0, odometry::INER_DIM, 1), numeric, analytic);
    REQUIRE( D.cwiseAbs().maxCoeff() < 1e-3 );
}

TEST_CASE( "tranformTo", "[EKF]" ) {
    const Eigen::MatrixXd P0 = load_csv<Eigen::MatrixXd>("test/data/P.csv");
    const Eigen::MatrixXd m0 = load_csv<Eigen::MatrixXd>("test/data/m.csv");

    odometry::Parameters params;
    params.odometry.cameraTrailLength = 5;
    params.odometry.hybridMapSize = 0;
    auto o = odometry::EKF::build(params);
    o->setState(m0);
    o->setStateCovariance(P0);

    constexpr int ANCHOR_IDX = 2;
    const Eigen::Vector3d pos0 = o->historyPosition(ANCHOR_IDX);
    const Eigen::Vector4d rot0 = o->historyOrientation(ANCHOR_IDX);

    const Eigen::Vector3d toPos = Eigen::Vector3d(0, 1, 0);
    const Eigen::Vector4d toRot = Eigen::Vector4d(1, 0, 0, 0);
    o->transformTo(toPos, toRot, ANCHOR_IDX);

    REQUIRE( (o->historyPosition(ANCHOR_IDX) - toPos).norm() < 1e-6 );
    REQUIRE( (o->historyOrientation(ANCHOR_IDX) - toRot).norm() < 1e-6 );

    o->transformTo(pos0, rot0, ANCHOR_IDX);

    REQUIRE( (o->getState() - m0).norm() < 1e-3 );
    REQUIRE( (o->getStateCovariance() - P0).norm() < 1e-3 );
}
