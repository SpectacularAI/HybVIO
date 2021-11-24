#ifndef UTIL_AUTO_HELP_PARSER_H
#define UTIL_AUTO_HELP_PARSER_H

#include <set>
#include <sstream>
#include <iomanip>
#include "parameter_parser.hpp"

/**
 * A helper class for generating a command line help text when the parameters
 * are accessed. This helps keeping these descriptions up to date with
 * minimal code overhead
 */
class AutoHelpParser : public ParameterParser {
private:
    std::ostringstream helpStream;
    std::set<std::string> missingKeys;

    template <class T> T parseWithHelp(const std::string &key, T def, const std::string &helpText, bool mandatory) {
        helpStream << std::setw(width) << ("-" + key)
            << std::setw(0) << "   " << helpText;
        if (mandatory)
            helpStream << " [mandatory]";
        else if (includeDefaults)
            helpStream << " [default " << def << "]";
        helpStream << std::endl;
        if (mandatory && !hasKey(key))
            missingKeys.insert(key);
        return getOrDefault<T>(key, def);
    }

public:
    bool includeDefaults = false;
    int width = 20;

    template <class T> T parseWithHelp(const std::string &key, T def, const std::string &helpText) {
        return parseWithHelp(key, def, helpText, false);
    }

    template <class T> T parseWithHelp(const std::string &key, const std::string &helpText) {
        return parseWithHelp(key, T{}, helpText, true);
    }

    std::string getHelpString() const {
        return helpStream.str();
    }

    void throwOnErrors() override {
        for (const auto &k : missingKeys) {
            throw std::runtime_error(k + " is a madatory parameter");
        }
        ParameterParser::throwOnErrors();
    }
};

#endif
