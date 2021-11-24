#include "parameter_parser.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "yaml-cpp/yaml.h"

#include "../util/logging.hpp"
#include "../util/util.hpp"

namespace {
// Remove spaces from beginning of string in-place.
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// Remove spaces from end of string in-place.
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Remove spaces from both ends of string in-place.
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

inline bool stobool(const std::string& str) {
    if (str == "true") return true;
    if (str == "false") return false;
    throw std::invalid_argument("invalid bool value " + str);

}
} // namespace

struct ParameterParser::impl {
    YAML::Node yamlConfig;
    std::map<std::string, std::string> keyValueMap;
    std::set<std::string> usedKeys;

    /**
     * Fetch a value by key, try to convert to wanted type and finally
     * mark the key as used. Throws a human-readable exception if
     * something goes wrong.
     */
    template <class T> T get(
        const std::string &key,
        const std::string &typeName,
        std::function<T(const std::string &)> converter) {

        // keyValueMap overrides yamlConfig
        const auto item = keyValueMap.find(key);
        if (item != keyValueMap.end()) {
            usedKeys.insert(key);
            try {
                return converter(item->second);
            } catch (const std::logic_error &e) {
                throw std::invalid_argument("invalid " + typeName + " value for "
                    + key + ": " + item->second);
            }
        }

        auto node = yamlConfig[key];
        if (node) {
            usedKeys.insert(key);
            return node.as<T>();
        }

        throw std::runtime_error("key " + key + " not found");
    }

    template <class T> std::vector<T> getList(
            const std::string &key,
            const std::string &typeName,
            std::function<T(const std::string &)> converter) {
        const auto item = keyValueMap.find(key);
        if (item != keyValueMap.end()) {
            usedKeys.insert(key);
            try {
                std::vector<T> values;
                util::splitWithMap(item->second, values, ',', converter);
                return values;
            } catch (const std::logic_error &e) {
                throw std::invalid_argument("invalid " + typeName + " value for std::vector<"
                                            + key + ">: " + item->second);
            }
        }

        auto node = yamlConfig[key];
        if (node) {
            usedKeys.insert(key);
            return node.as<std::vector<T>>();
        }

        throw std::runtime_error("key " + key + " not found");
    }
};

ParameterParser::ParameterParser() : pimpl(new impl()) {}
ParameterParser::~ParameterParser() = default;

void ParameterParser::add(const std::string &key, const std::string &value) {
    assert(pimpl->keyValueMap.count(key) == 0);
    pimpl->keyValueMap[key] = value;
}

std::set<std::string> ParameterParser::getUnusedKeys() const {
    std::set<std::string> unused;
    for (const auto &item : pimpl->keyValueMap) {
        if (pimpl->usedKeys.count(item.first) == 0) unused.insert(item.first);
    }
    for(YAML::const_iterator it = pimpl->yamlConfig.begin(); it != pimpl->yamlConfig.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (pimpl->usedKeys.count(key) == 0) unused.insert(key);
    }
    return unused;
}

bool ParameterParser::hasKey(const std::string &key) const {
    return pimpl->keyValueMap.count(key) > 0 || pimpl->yamlConfig[key];
}

template <> int ParameterParser::get<int>(const std::string &key) {
    return pimpl->get<int>(key, "integer", [](const std::string &k) { return std::stoi(k); });
}
template <> float ParameterParser::get<float>(const std::string &key) {
    return pimpl->get<float>(key, "float", [](const std::string &k) { return std::stof(k); });
}
template <> double ParameterParser::get<double>(const std::string &key) {
    return pimpl->get<double>(key, "double", [](const std::string &k) { return std::stod(k); });
}
template <> bool ParameterParser::get<bool>(const std::string &key) {
    return pimpl->get<bool>(key, "boolean", [](const std::string &k) {
        // Interpret, e.g, "-x" as "-x=true".
        return k.empty() ? true : stobool(k);
    });
}
template <> std::string ParameterParser::get<std::string>(const std::string &key) {
    return pimpl->get<std::string>(key, "string", [](const std::string &k) { return k; });
}
template <> unsigned ParameterParser::get<unsigned>(const std::string &key) {
    return pimpl->get<unsigned>(key, "unsigned", [](const std::string &k) { return std::stoi(k); });
}
template <> std::vector<std::string> ParameterParser::get<std::vector<std::string>>(const std::string &key) {
    return pimpl->getList<std::string>(key, "string", [](const std::string &k) { return k; });
}
template <> std::vector<int> ParameterParser::get<std::vector<int>>(const std::string &key) {
    return pimpl->getList<int>(key, "int", [](const std::string &k) { return std::stoi(k); });
}
template <> std::vector<double> ParameterParser::get<std::vector<double>>(const std::string &key) {
    return pimpl->getList<double>(key, "double", [](const std::string k) { return std::stod(k); });
}

void ParameterParser::parseDelimited(std::istream &values, char groupDelimiter, char valueDelimiter) {
    std::string line;
    while (std::getline(values, line, groupDelimiter)) {
        trim(line);
        if (line == "") continue;
        std::vector<std::string> tokens;
        util::splitWithMap<std::string>(line, tokens, valueDelimiter, [](const std::string &k) { return k; });
        if (tokens.size() != 2) {
            throw std::runtime_error("Could not parse group: " + line);
        }
        add(tokens[0], tokens[1]);
    }
}

void ParameterParser::parseDelimited(const std::string &string, char groupDelimiter, char valueDelimiter) {
    std::istringstream iss(string);
    parseDelimited(iss, groupDelimiter, valueDelimiter);
}

void ParameterParser::parseCommandLine(int argc, char *argv[]) {
    std::string param;
    for (int i=1; i<argc; ++i) {
        const std::string arg(argv[i]);
        // remove any dashes before the parameter name
        std::size_t nDashes = 0;
        while (nDashes < arg.size() && arg[nDashes] == '-') nDashes++;
        param = arg.substr(nDashes);
        if (param.empty() || nDashes < 1)
            throw std::runtime_error("invalid parameter " + arg);
        auto eqIdx = param.find('=');
        std::string value = "";
        if (eqIdx != std::string::npos) {
            value = param.substr(eqIdx+1);
            param = param.substr(0, eqIdx);
        }
        add(param, value);
    }
}

void ParameterParser::parseJson(std::istream &stream) {
    nlohmann::json j;
    stream >> j;
    for (nlohmann::json::iterator it = j.begin(); it != j.end(); ++it) {
        add(it.key(), it.value());
    }
}

void ParameterParser::parseYaml(std::istream &stream) {
    // TOOD: Simple top level merge. Doesn't work if we add hierarchy and want to merge maps withing config files.
    auto newConfig = YAML::Load(stream);
    for(YAML::const_iterator it = newConfig.begin(); it != newConfig.end(); ++it) {
        pimpl->yamlConfig[it->first.as<std::string>()] = it->second;
    }
}

void ParameterParser::throwOnErrors() {
    if (!getUnusedKeys().empty()) {
        std::ostringstream unusedMsg;
        unusedMsg << "unused key(s):";
        for (const auto &key : getUnusedKeys()) {
            unusedMsg << " " << key;
        }
        throw std::runtime_error(unusedMsg.str());
    }
}
