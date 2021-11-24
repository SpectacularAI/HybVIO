#include "util.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>

namespace util {

#ifdef USE_LOGURU
void setup_logging(int level) {
    // see https://github.com/emilk/loguru/blob/f64f3fc088392869142c7b170dbb015c2f79bd5a/loguru.hpp#L311
    loguru::g_stderr_verbosity = level;
    loguru::g_preamble_header = false;
    loguru::g_preamble_date = false;
    loguru::g_preamble_time = false;
    loguru::g_preamble_uptime = false;
    loguru::g_preamble_thread = false;
    loguru::g_preamble_file = true;
    loguru::g_preamble_verbose = false;
    loguru::g_preamble_pipe = false;

    // Do not use actual `argv` because loguru's parsing interferes
    // with our own commandline parsing functions.
    int argc = 1;
    char *argv[] = {(char *)"odometry", NULL};
    loguru::init(argc, argv);
}
#else
void setup_logging(int level) {
    (void)level;
}
#endif

// Modulo operation `l % r` that returns positive remainder when `l` is negative.
unsigned modulo(int l, unsigned r) {
    int mod = l % (int)r;
    if (mod < 0) {
        mod += r;
    }
    return mod;
}

/**
 * Helper: remove a short suffix such as .csv, .mov or .jpeg
 * from a file name
 */
std::string removeFileSuffix(const std::string &s) {
    const auto idx = s.rfind(".");
    if (idx == std::string::npos) return s;
    const auto base = s.substr(0, idx);
    const auto suffix = s.substr(idx + 1);
    if (suffix.size() <= 5) return base;
    return s;
}

std::string getFileUnixPath(const std::string &s) {
    const auto idx = s.rfind("/");
    if (idx == std::string::npos) return s;
    return s.substr(0, idx);
}

std::string stripTrailingSlash(const std::string &s) {
    if (s.empty()) return s;
    return *s.rbegin() == '/' ? s.substr(0, s.size()-1) : s;
}

std::string joinUnixPath(const std::string &path, const std::string &fn) {
    assert(path.size() > 0);
    return stripTrailingSlash(path) + "/" + fn;
}

std::string toUpper(const std::string &in) {
    std::string out = in;
    std::transform(out.begin(), out.end(), out.begin(), ::toupper);
    return out;
}

} // namespace util
