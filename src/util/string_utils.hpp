#ifndef UTIL_STRING_UTIL_H_
#define UTIL_STRING_UTIL_H_

#include <string>
#include <Eigen/StdVector>

namespace util {

template <class eigenType> std::string eigenToMultilineString(eigenType x) {
    std::stringstream ss;
    ss << x;
    auto s = ss.str();
    return s;
}

template <class eigenType> std::string eigenToString(eigenType x) {
    auto s = eigenToMultilineString(x);
    std::replace(s.begin(), s.end(), '\n', ',');
    return s;
}

} // namespace util

#endif
