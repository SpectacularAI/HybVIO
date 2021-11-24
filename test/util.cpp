#include "catch2/catch.hpp"

#include "../src/odometry/util.hpp"
#include "../src/api/type_convert.hpp"
#include "helpers.hpp"

#include <iostream>

TEST_CASE( "quat2rmat", "[util]" ) {
    // Matlab (official funcs): q = rotm2quat(rotx(15)*roty(3)*rotz(-5));
    Eigen::Vector4d q; q << 0.990310843256666, 0.129225220713441, 0.031619820909086, -0.039817872689419;

    // Matlab (our func): rmat_e = util::quat2rmat(q);
    Eigen::Matrix3d rmat_e; rmat_e <<
         0.994829447880333, 0.087036298831283,  0.052335956242944,
        -0.070691985487699, 0.963430758692103, -0.258464342596353,
        -0.072917849789463, 0.253428206582672,  0.964602058514480;

    Eigen::Matrix3d rmat = odometry::util::quat2rmat(q);
    REQUIRE( (rmat - rmat_e).array().abs().sum() < 1e-5 );

    // Eigen also has an equivalent function, but it takes care to use the Quaterniond type since their
    // internal representation is (x, y, z, w) while we use (w, x, y, z).
    // See: https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html#ad30f4da9a2c0c8dd95520ee8a6d14ef6
    Eigen::Quaterniond q_eigen(q[0], q[1], q[2], q[3]);
    // Eigen::Quaterniond q_eigen(q); // This is wrong.
    Eigen::Matrix3d rmat_eigen = q_eigen.toRotationMatrix();
    REQUIRE( (rmat_eigen - rmat_e).array().abs().sum() < 1e-5 );
}

TEST_CASE( "quat2rmat_d", "[util]" ) {
    // Matlab (official funcs): q = rotm2quat(rotx(15)*roty(3)*rotz(-5));
    Eigen::Vector4d q; q << 0.990310843256666, 0.129225220713441, 0.031619820909086, -0.039817872689419;
    Eigen::Matrix3d dR_e[4];
    dR_e[0] <<
        1.980621686513332,   0.079635745378838,   0.063239641818172,
        -0.079635745378838,   1.980621686513332,  -0.258450441426882,
        -0.063239641818172,   0.258450441426882,   1.980621686513332;
    dR_e[1] <<
        0.258450441426882,   0.063239641818172,  -0.079635745378838,
        0.063239641818172,  -0.258450441426882,  -1.980621686513332,
        -0.079635745378838,   1.980621686513332,  -0.258450441426882;
    dR_e[2] <<
        -0.063239641818172,   0.258450441426882,   1.980621686513332,
        0.258450441426882,   0.063239641818172,  -0.079635745378838,
        -1.980621686513332,  -0.079635745378838,  -0.063239641818172;
    dR_e[3] <<
        0.079635745378838,  -1.980621686513332,   0.258450441426882,
        1.980621686513332,   0.079635745378838,   0.063239641818172,
        0.258450441426882,   0.063239641818172,  -0.079635745378838;

    Eigen::Matrix3d dR[4];
    odometry::util::quat2rmat_d(q, dR);
    for (int i = 0; i < 4; i++) {
        REQUIRE( (dR[i] - dR_e[i]).array().abs().sum() < 1e-5 );
    }
}

TEST_CASE( "rmat2quat", "[util]" ) {
    // Matlab (official funcs): q_e = rotm2quat(rotx(15)*roty(3)*rotz(-5));
    Eigen::Vector4d q_e; q_e << 0.990310843256666, 0.129225220713441, 0.031619820909086, -0.039817872689419;

    // Matlab (our func): rmat = util::quat2rmat(q_e);
    Eigen::Matrix3d rmat; rmat <<
         0.994829447880333, 0.087036298831283,  0.052335956242944,
        -0.070691985487699, 0.963430758692103, -0.258464342596353,
        -0.072917849789463, 0.253428206582672,  0.964602058514480;

    Eigen::Vector4d q = odometry::util::rmat2quat(rmat);
    REQUIRE( (q - q_e).array().abs().sum() < 1e-5 );
}

TEST_CASE( "stdev", "[util]" ) {
    double std_e = 1.87082869338697;
    SECTION( "VectorXd" ) {
        Eigen::VectorXd v(6); v << -1.0, 0.0, 1.0, 2.0, 3.0, 4.0;
        REQUIRE( std::abs(odometry::util::stdev(v) - std_e) < 1e-8);
    }
    SECTION( "MatrixN1" ) {
        Eigen::Matrix<double, 6, 1> v; v << -1.0, 0.0, 1.0, 2.0, 3.0, 4.0;
        REQUIRE( std::abs(odometry::util::stdev(v) - std_e) < 1e-8);
    }
    SECTION( "Matrix1N" ) {
        Eigen::Matrix<double, 1, 6> v; v << -1.0, 0.0, 1.0, 2.0, 3.0, 4.0;
        REQUIRE( std::abs(odometry::util::stdev(v) - std_e) < 1e-8);
    }
    SECTION( "MatrixXd" ) {
        Eigen::MatrixXd v(6, 1); v << -1.0, 0.0, 1.0, 2.0, 3.0, 4.0;
        REQUIRE( std::abs(odometry::util::stdev(v) - std_e) < 1e-8);
    }
    SECTION( "VectorXd single" ) {
        Eigen::VectorXd v(1); v << 4.0;
        REQUIRE( std::abs(odometry::util::stdev(v) - 0.0) < 1e-8);
    }
}

TEST_CASE( "cond", "[util]" ) {
    // These functions are not numerically same as the Matlab functions.
    // Matlab: A = hilb(10);
    Eigen::MatrixXd A = load_csv<Eigen::MatrixXd>("test/data/hilb10.csv");
    REQUIRE( odometry::util::rcond(A) < 1e-10);
    REQUIRE( odometry::util::rcond_ldlt(A) < 1e-10);
    REQUIRE( odometry::util::cond(A) > 1e9);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
    REQUIRE( odometry::util::rcond(I) == 1);
    REQUIRE( odometry::util::rcond_ldlt(I) == 1);
    REQUIRE( odometry::util::cond(I) == 1);
}
