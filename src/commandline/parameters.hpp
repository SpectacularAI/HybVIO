#ifndef COMMANDLINE_PARAMETERS_HPP
#define COMMANDLINE_PARAMETERS_HPP

#include <iostream>
#include "../odometry/parameters.hpp"
#include "codegen/output/cmd_parameters.hpp"
#include "../util/parameter_parser.hpp"
#include <iomanip>
#include <nlohmann/json.hpp>

struct CommandLineParameters {
    odometry::Parameters parameters;
    cmd::Parameters cmd;

    CommandLineParameters(int argc, char *argv[]) {
        std::ifstream cmdParametersFile("../data/cmd.json");
        if (cmdParametersFile.is_open()) {
            parse_cmd_parameters(cmdParametersFile);
        }

        parse_argv(argc, argv);
    }

    void parse_argv(int argc, char *argv[]) {
        ParameterParser parser;
        parser.parseCommandLine(argc, argv);

        odometry::setParsedParameters(parameters, parser);
        cmd::setParsedParameters(cmd, parser);

        if (parser.hasKey("h") || parser.hasKey("help")) {
            std::cout << "Supported arguments" << std::endl;
            for (const cmd::Help &help : cmd::HELPS) {
                std::cout
                    << std::setw(25) << ("-" + (!help.shortName.empty() ? help.shortName : help.name))
                    << std::setw(0) << "   " << help.doc
                    << " [" << help.defaultValue << "]"
                    << std::endl;
            }
            std::cout << "\n ... or any of the values in codegen/parameter_definitions.c\n"
                << " without the leading namespace, e.g., -maxSuccessfulVisualUpdates=10\n"
                << std::endl;
            exit(0);
        }

        parser.throwOnErrors();
    }

    void parse_calibration_json(std::istream &stream) {
        using json = nlohmann::json;
        json config = json::parse(stream);
        auto cameras = config["cameras"].get<json>();
        for (size_t i = 0; i < cameras.size() && i <= 2; i++) {
            json &camera = cameras[i];
            if (camera.find("imuToCamera") != camera.end()) {
                std::vector<double> imuToCameraMatrix;
                for (int c = 0; c < 4; c++)
                    for (int r = 0; r < 4; r++)
                        imuToCameraMatrix.push_back(camera["imuToCamera"][r][c].get<double>());
                if (i == 0)
                    parameters.odometry.imuToCameraMatrix = imuToCameraMatrix;
                else
                    parameters.odometry.secondImuToCameraMatrix = imuToCameraMatrix;
            }
            if (camera["model"].get<std::string>() == "kannala-brandt4") {
                parameters.tracker.fisheyeCamera = true;
            }

            #define setDoubleIfExists(keyName, param) \
                if (camera.find(keyName) != camera.end()) param = camera[keyName].get<double>();
            #define setVectorIfExists(keyName, param) \
                if (camera.find(keyName) != camera.end()) param = camera[keyName].get<std::vector<double>>();

            if (i == 0) {
                setDoubleIfExists("focalLengthX", parameters.tracker.focalLengthX)
                setDoubleIfExists("focalLengthY", parameters.tracker.focalLengthY)
                setDoubleIfExists("principalPointX", parameters.tracker.principalPointX)
                setDoubleIfExists("principalPointY", parameters.tracker.principalPointY)
                // Both singular and plural forms have been in use.
                setVectorIfExists("distortionCoefficients", parameters.tracker.distortionCoeffs)
                setVectorIfExists("distortionCoefficient", parameters.tracker.distortionCoeffs)
            } else {
                setDoubleIfExists("focalLengthX", parameters.tracker.secondFocalLengthX)
                setDoubleIfExists("focalLengthY", parameters.tracker.secondFocalLengthY)
                setDoubleIfExists("principalPointX", parameters.tracker.secondPrincipalPointX)
                setDoubleIfExists("principalPointY", parameters.tracker.secondPrincipalPointY)
                setVectorIfExists("distortionCoefficients", parameters.tracker.secondDistortionCoeffs)
                setVectorIfExists("distortionCoefficient", parameters.tracker.secondDistortionCoeffs)
            }
        }
    }

    void parse_algorithm_parameters(std::istream &parametersFile) {
        ParameterParser parser;
        parser.parseDelimited(parametersFile);

        odometry::setParsedParameters(parameters, parser);
        parser.throwOnErrors();
    }

    void parse_yaml_config(std::istream &yamlConfigFile) {
        ParameterParser parser;
        parser.parseYaml(yamlConfigFile);
        odometry::setParsedParameters(parameters, parser);
        parser.throwOnErrors();
    }

    void parse_cmd_parameters(std::istream &parametersStream) {
        ParameterParser parser;
        parser.parseJson(parametersStream);

        // Only apply to cmd parameters. Algorithm parameters if forgotten
        // in the fixed data/cmd.json could cause confusion.
        cmd::setParsedParameters(cmd, parser);
        parser.throwOnErrors();
    }
};

#endif
