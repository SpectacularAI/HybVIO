#include "input.hpp"

#include "../util/util.hpp"

namespace odometry {

bool pathHasFile(const std::string &path, const std::string &file) {
    // `path` might not be folder but we want failure in that case anyway.
    std::string dataPath = util::joinUnixPath(path, file);
    std::ifstream dataFile(dataPath);
    return dataFile.is_open();
}

std::string getParametersStringWithPath(const std::string &parametersPath) {
    if (parametersPath.empty()) return "";
    std::ifstream paramFile(parametersPath);
    assert(paramFile.is_open());
    std::string s, line;
    int i = 0;
    while (std::getline(paramFile, line)) {
        if (i > 0) s += "\n";
        s += line;
        ++i;
    }
    return s;
}

} // namespace odometry
