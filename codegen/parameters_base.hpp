#ifndef ODOMETRY_PARAMETERS_H_
#define ODOMETRY_PARAMETERS_H_

#include <istream>
#include <vector>
#include <Eigen/Dense>

class ParameterParser; // fwd decl

namespace odometry {

// CODEGEN-HPP-TOP

// CODEGEN-HPP-STRUCT

struct Parameters {
    enum VerbosityLevel {
        // SILENT = 0 // not supported yet
        VERBOSITY_INFO = 1,
        VERBOSITY_DEBUG = 2,
        VERBOSITY_EXTRA = 3
    };
    int verbosity;
    Eigen::Matrix4d imuToCamera;
    Eigen::Matrix4d secondImuToCamera;
    Eigen::Matrix4d imuToOutput;

    // CODEGEN-HPP-SUB-STRUCT

    Parameters(); // Default parameters
};

void setParameterString(Parameters& p, std::istream& values);
void setParsedParameters(Parameters& p, ParameterParser& parser);

} // namespace odometry

#endif
