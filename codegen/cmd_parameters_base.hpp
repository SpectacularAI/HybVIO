#ifndef CMD_PARAMETERS_H_
#define CMD_PARAMETERS_H_

#include <istream>
#include <vector>

#include "parameters.hpp"

class ParameterParser; // fwd decl

namespace cmd {

// CODEGEN-HPP-TOP

// CODEGEN-HPP-STRUCT

struct Parameters {
    // CODEGEN-HPP-SUB-STRUCT

    Parameters(); // Default parameters
};

struct Help {
    std::string name;
    std::string shortName;
    std::string defaultValue;
    std::string doc;
};

extern std::vector<Help> HELPS;

void setParsedParameters(Parameters& p, ParameterParser& parser);

} // namespace cmd

#endif
