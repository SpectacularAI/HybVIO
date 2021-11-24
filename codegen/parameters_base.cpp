#include "parameters.hpp"
#include "../../src/util/parameter_parser.hpp"
#include "../../src/util/util.hpp"
#include "../../src/util/logging.hpp"
#include "../../src/odometry/util.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>

namespace odometry {

Parameters::Parameters() :
    verbosity(Parameters::VERBOSITY_INFO),
    imuToCamera(Eigen::Matrix4d::Zero()),
    secondImuToCamera(Eigen::Matrix4d::Zero()),
    imuToOutput(Eigen::Matrix4d::Identity()),
    // CODEGEN-CPP-SUB-STRUCT
{}

// CODEGEN-CPP-STRUCT

// Example input format: "useSlam true; maxVisualUpdates 3"
void setParameterString(Parameters& p, std::istream& values) {
    ParameterParser parser;
    parser.parseDelimited(values);
    setParsedParameters(p, parser);
    parser.throwOnErrors();
}

void setParsedParameters(Parameters& p, ParameterParser& parser) {
    // CODEGEN-CPP-SET-PARAMETER

    if (parser.hasKey("videoRotation")) {
        // NOTE: here's a gotcha... If you set videoRotation in both cmdline
        // and parameters.txt, they cumulate! so CW90 in both = CW180
        // this is not ideal
        Eigen::Matrix2d imgRot;
        const auto value = parser.get<std::string>("videoRotation");
        if (value == "NONE") {
            imgRot = Eigen::Matrix2d::Identity();
        }
        else if (value == "CW90") {
            imgRot <<  0, 1, -1, 0;
        }
        else if (value == "CW180") {
            imgRot = -Eigen::Matrix2d::Identity();
        }
        else if (value == "CW270") {
            imgRot << 0, -1, 1, 0;
        }
        else {
            throw std::invalid_argument("Unknown videoRotation parameter: " + value);
        }

        auto &imuToCamVec = p.odometry.imuToCameraMatrix;
        Eigen::Matrix4d imuToCam = odometry::util::vec2matrix(imuToCamVec);
        const Eigen::Matrix2d oldSubmat = imuToCam.topLeftCorner<2, 2>();
        imuToCam.topLeftCorner<2, 2>() = imgRot * oldSubmat;
        imuToCamVec.resize(4*4);
        Eigen::Map<Eigen::Matrix4d>(imuToCamVec.data()) = imuToCam;
    }
}

} // namespace odometry
