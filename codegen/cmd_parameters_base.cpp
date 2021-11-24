#include "cmd_parameters.hpp"
#include "../../src/util/parameter_parser.hpp"
#include "../../src/util/util.hpp"
#include "../../src/util/logging.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace cmd {

Parameters::Parameters() :
    // CODEGEN-CPP-SUB-STRUCT
{}

// CODEGEN-CPP-STRUCT

void setParsedParameters(Parameters& p, ParameterParser& parser) {
    // CODEGEN-CPP-SET-PARAMETER
}

std::vector<Help> HELPS = {
    // CODEGEN-CPP-HELP
};

} // namespace cmd
