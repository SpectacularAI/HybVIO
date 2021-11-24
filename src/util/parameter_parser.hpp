#ifndef UTIL_PARAMETER_PARSER_H
#define UTIL_PARAMETER_PARSER_H

#include <string>
#include <memory>
#include <set>
#include <istream>

class ParameterParser {
public:
    ParameterParser();
    ~ParameterParser();

    void add(const std::string &key, const std::string &value);
    void parseDelimited(std::istream &stream, char groupDelimiter=';', char valueDelimiter=' ');
    void parseDelimited(const std::string &string, char groupDelimiter=';', char valueDelimiter=' ');
    void parseCommandLine(int argc, char *argv[]);
    void parseJson(std::istream &stream);
    // Note: YAML is always overriden by configurations from other sources like JSON and CommandLine
    void parseYaml(std::istream &stream);

    template <class T> T get(const std::string &key);
    template <class T> inline T getOrDefault(const std::string &key, T def) {
        return hasKey(key) ? get<T>(key) : def;
    }
    bool hasKey(const std::string &key) const;
    std::set<std::string> getUnusedKeys() const;
    virtual void throwOnErrors();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

#endif
