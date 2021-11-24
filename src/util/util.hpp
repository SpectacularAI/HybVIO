#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace util {

void setup_logging(int level);
unsigned modulo(int l, unsigned r);
std::string removeFileSuffix(const std::string &s);
std::string getFileUnixPath(const std::string &s);
std::string stripTrailingSlash(const std::string &s);
std::string joinUnixPath(const std::string &path, const std::string &fn);
std::string toUpper(const std::string &in);

// Split a string into vector of strings using a delimiter charactes.
template <typename T> void splitWithMap(const std::string& str, std::vector<T>& tokens, char delim, std::function<T(const std::string &)> mapping) {
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(mapping(token));
    }
}

// Based on <https://en.cppreference.com/w/cpp/algorithm/random_shuffle>
//
// Intended to replace `std::shuffle()` for algorithmic purposes as that function
// and `std::uniform_int_distribution` may not be deterministic across machines.
//
// From the page:
// “Note that the implementation is not dictated by the standard, so even if you use
// exactly the same RandomFunc or URBG (Uniform Random Number Generator) you may get
// different results with different standard library implementations. ”
template<class RandomIt, class URBG>
void shuffleDeterministic(RandomIt first, RandomIt last, URBG &&g) {
    int n = last - first;
    for (int i = n - 1; i > 0; --i) {
        std::swap(first[i], first[g() % (i + 1)]);
    }
}

} // namespace util

#endif // UTIL_UTIL_H_
