// Read sensor data and other input mainly from a single JSONL file. Supports
// a folder containing `data.jsonl`, video and metadata (same as the newer
// CSV format with the sensor data file replaced).

#include "input.hpp"
#include "parameters.hpp"
#include "../util/util.hpp"
#include "../util/logging.hpp"
#include "../util/gps.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace odometry;
using json = nlohmann::json;

bool replace(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string findVideoSuffix(std::string videoPathNoSuffix) {
    for (std::string suffix : { "mov", "avi", "mp4" }) {
        const auto path = videoPathNoSuffix + "." + suffix;
        std::ifstream testFile(videoPathNoSuffix + "." + suffix);
        if (testFile.is_open()) return path;
    }
    throw std::runtime_error("Could not find any video " + videoPathNoSuffix + ".mov|avi|mp4");
}

class InputJSONL : public InputI {
public:
    InputJSONL(const std::string &inputFolderPath, bool requireAllFiles, const std::string &parametersPathCustom);
    InputSample nextType() final;
    void getGyroscope(double &t, api::Vector3d &p) final;
    void getAccelerometer(double &t, api::Vector3d &p) final;
    void getFrames(double &t, int &framesInd, std::vector<InputFrame> &frames) final;
    std::string getInputVideoPath(int cameraInd) const final;
    void setAlgorithmParametersFromData(odometry::Parameters &parameters) final;
    void setParameters(CommandLineParameters &parameters) final;
    std::string getParametersString() const final;
    bool getParametersAvailable() const final;
    std::map<api::PoseHistory, std::vector<api::Pose>> getPoseHistories() final;
    std::string getLastJSONL() const final;
    bool canEcho() const final;
private:
    bool echo(const json &j);
    api::Pose readGps(const json &j, const std::string &field);

    std::string inputFolderPath;
    std::string inputVideoPath;
    std::string parametersPath;
    std::string dataPath;
    std::ifstream dataFile;
    bool parametersAvailable;
    std::string line;
    double t;
    std::array<double, 3> sensorValues;
    std::map<int, InputFrame> frames;
    int framesInd;
    util::GpsToLocalConverter gpsToLocal;
};

bool isFileReadable(std::string filePath) {
    std::ifstream parametersFile(filePath);
    if (parametersFile.is_open()) {
        parametersFile.close();
        return true;
    }
    return false;
}

bool isYamlFile(std::string filePath) {
    std::string suffix = ".yaml";
    if (suffix.size() > filePath.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), filePath.rbegin());
}

InputJSONL::InputJSONL(
    const std::string &inputFolderPath,
    bool requireAllFiles,
    const std::string &parametersPathCustom
) :
    inputFolderPath(inputFolderPath),
    parametersAvailable(false),
    t(0.0)
{
    dataPath = util::joinUnixPath(inputFolderPath, "data.jsonl");
    dataFile.open(dataPath);
    if (!dataFile.is_open()) {
        log_error("Could not open %s.", dataPath.c_str());
        assert(false);
    }

    if (requireAllFiles) {
        parametersPath = util::joinUnixPath(inputFolderPath, "vio_config.yaml");
        if (!isFileReadable(parametersPath)) {
            parametersPath = util::joinUnixPath(inputFolderPath, "parameters.txt");
        }
        inputVideoPath = findVideoSuffix(util::joinUnixPath(inputFolderPath, "data"));
    }
    if (!parametersPathCustom.empty()) {
        log_info("Loaded custom parameters from `%s`.", parametersPathCustom.c_str());
        parametersPath = parametersPathCustom;
    }

    if (isFileReadable(parametersPath)) {
        parametersAvailable = true;
    } else {
        log_warn("Could not find `vio_config.yaml` or `parameters.txt`.");
        parametersPath.clear();
    }
}

InputSample InputJSONL::nextType() {
    while (std::getline(dataFile, line)) {
        json j = json::parse(line);
        if (j.find("sensor") != j.end()) {
            t = j["time"].get<double>();
            sensorValues = j["sensor"]["values"];
            std::string sensorType = j["sensor"]["type"];
            if (sensorType == "gyroscope") {
                return InputSample::GYROSCOPE;
            }
            else if (sensorType == "accelerometer") {
                return InputSample::ACCELEROMETER;
            }
            else {
                log_debug("Unknown sensor type: %s", sensorType.c_str());
            }
        }
        else if (j.find("frames") != j.end()) {
            frames.clear();
            json jFrames = j["frames"];
            for (json::iterator jFrame = jFrames.begin(); jFrame != jFrames.end(); ++jFrame) {
                InputFrame frame = {
                    .intrinsic = api::CameraParameters(),
                    .t = (*jFrame)["time"].get<double>()
                };
                if (!(*jFrame)["cameraParameters"].is_null()) {
#define X(FIELD) \
                    if (!(*jFrame)["cameraParameters"][#FIELD].is_null()) { \
                        frame.intrinsic.FIELD = (*jFrame)["cameraParameters"][#FIELD].get<double>(); \
                    }
                    X(focalLengthX)
                    X(focalLengthY)
                    X(principalPointX)
                    X(principalPointY)
#undef X
                    bool hasDirFocal = frame.intrinsic.focalLengthX > 0.0 && frame.intrinsic.focalLengthY > 0.0;
                    if (!hasDirFocal && !(*jFrame)["cameraParameters"]["focalLength"].is_null()) {
                        double focalLength = (*jFrame)["cameraParameters"]["focalLength"].get<double>();
                        frame.intrinsic.focalLengthX = focalLength;
                        frame.intrinsic.focalLengthY = focalLength;
                    }
                }
                int cameraInd = (*jFrame)["cameraInd"].get<int>();
                // Use map to allow any order of cameraInds in the JSON array.
                frames.insert({cameraInd, frame});
            }
            if (!frames.empty()) {
                framesInd = j["number"].get<int>();
                return InputSample::FRAME;
            }
            log_debug("No frames in frame group.");
        }
        else if (echo(j)) {
            return InputSample::ECHO_RECORDING;
        }
    }
    return InputSample::NONE;
}

bool InputJSONL::echo(const json &j) {
    // Sample types that will be echoed and their replacement name if any.
    static const std::tuple<std::string, std::string> echoed[] = {
        { "groundTruth", "" },
        { "ARKit", "" },
        { "arengine", "" },
        { "arcore", "" },
        { "realsense", "" },
        { "gps", "" },
        { "rtkgps", "" },
        { "output", "outputPrevious" }
    };
    for (const auto &e : echoed) {
        if (j.find(std::get<0>(e)) != j.end()) {
            if (!std::get<1>(e).empty()) {
                replace(line, std::get<0>(e), std::get<1>(e));
            }
            return true;
        }
    }
    return false;
}

void InputJSONL::getGyroscope(double &_t, api::Vector3d &_p) {
    _t = t;
    _p.x = sensorValues[0];
    _p.y = sensorValues[1];
    _p.z = sensorValues[2];
}

void InputJSONL::getAccelerometer(double &_t, api::Vector3d &_p) {
    _t = t;
    _p.x = sensorValues[0];
    _p.y = sensorValues[1];
    _p.z = sensorValues[2];
}

void InputJSONL::getFrames(double &t, int &framesInd, std::vector<InputFrame> &frames) {
    assert(!this->frames.empty());
    size_t n = this->frames.size();
    frames.clear();
    // Assumes the keys of `this->frames` are successive without gaps.
    for (size_t i = 0; i < n; ++i) {
        frames.push_back(this->frames.at(i));
    }
    t = frames[0].t;
    framesInd = this->framesInd;
}

std::string InputJSONL::getInputVideoPath(int cameraInd) const {
    if (cameraInd > 0) {
        return findVideoSuffix(util::joinUnixPath(inputFolderPath, "data" + std::to_string(cameraInd + 1)));
    }
    else {
        return inputVideoPath;
    }
}

void InputJSONL::setAlgorithmParametersFromData(odometry::Parameters &parameters) {
    std::ifstream dataFile2;
    dataFile2.open(dataPath);
    while (std::getline(dataFile2, line)) {
        json j = json::parse(line);
        if (j.find("model") != j.end()
                && j["model"].get<std::string>().find("KANNALA_BRANDT4") != std::string::npos) {
            std::vector<double> coeffs = j["coeffs"].get<std::vector<double>>();
            // Some RealSense data erroneously sets a fifth coeff with 0 value.
            assert(coeffs.size() >= 4);
            coeffs.resize(4);

            parameters.tracker.fisheyeCamera = true;
            int cameraInd = j["cameraInd"].get<int>();
            if (cameraInd == 0) {
                parameters.tracker.distortionCoeffs = coeffs;
            }
            else if (cameraInd == 1) {
                parameters.tracker.secondDistortionCoeffs = coeffs;
            }
            log_debug("Set fisheye cam%d: [%.3f, %.3f, %.3f, %.3f]",
                cameraInd, coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
        }

        if (j.find("imuToCamera") != j.end()) {
            std::vector<double> v;
            try {
                // Read flattened column-major.
                v = j["imuToCamera"].get<std::vector<double>>();
            } catch (json::type_error& e) {
                // Convert from nested row-major representation to flattened column-major.
                std::vector<std::vector<double>> M = j["imuToCamera"].get<std::vector<std::vector<double>>>();
                size_t n = M.size();
                assert(n == 3 || n == 4);
                for (size_t i = 0; i < n; ++i) {
                    assert(M[i].size() == n);
                    for (size_t j = 0; j < n; ++j) {
                        v.push_back(M[j][i]);
                    }
                }
            }
            assert(v.size() == 9 || v.size() == 16);

            int cameraInd = j["cameraInd"].get<int>();
            if (cameraInd == 0) {
                parameters.odometry.imuToCameraMatrix = v;
            }
            else if (cameraInd == 1) {
                parameters.odometry.secondImuToCameraMatrix = v;
            }
        }
    }
}

std::map<api::PoseHistory, std::vector<api::Pose>> InputJSONL::getPoseHistories() {
    std::map<api::PoseHistory, std::vector<api::Pose>> poseHistories;

    static const std::tuple<std::string, api::PoseHistory> historyTypes[] = {
        { "groundTruth", api::PoseHistory::GROUND_TRUTH },
        { "ARKit", api::PoseHistory::ARKIT },
        { "arengine", api::PoseHistory::ARENGINE },
        { "arcore", api::PoseHistory::ARCORE },
        { "realsense", api::PoseHistory::REALSENSE },
        { "zed", api::PoseHistory::ZED },
        { "output", api::PoseHistory::OUR_PREVIOUS }
    };

    std::ifstream dataFile2;
    dataFile2.open(dataPath);
    while (std::getline(dataFile2, line)) {
        json j = json::parse(line);
        for (const auto &historyType : historyTypes) {
            if (j.find(std::get<0>(historyType)) != j.end()) {
                try {
                    json jPose = j[std::get<0>(historyType)];
                    api::Pose pose = {
                        .time = j["time"].get<double>(),
                        .position = {
                            .x = jPose["position"]["x"],
                            .y = jPose["position"]["y"],
                            .z = jPose["position"]["z"]
                        },
                        .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 }
                    };
                    if (jPose.find("orientation") != jPose.end()) {
                        pose.orientation = {
                            // Conjugate because the JSONL format is documented as device-to-world,
                            // whereas in our code api::Pose orientations are world-to-device.
                            .x = -static_cast<double>(jPose["orientation"]["x"]),
                            .y = -static_cast<double>(jPose["orientation"]["y"]),
                            .z = -static_cast<double>(jPose["orientation"]["z"]),
                            .w = jPose["orientation"]["w"]
                        };
                    }
                    poseHistories[std::get<1>(historyType)].push_back(pose);
                } catch (json::type_error& e) {
                    log_debug("Invalid input in pose history, skipping it: %s, exception id %d", e.what(), e.id);
                    log_debug("  %s", line.c_str());
                }
                break;
            }
        }
        if (j.find("gps") != j.end()) {
            poseHistories[api::PoseHistory::GPS].push_back(readGps(j, "gps"));
        }
        else if (j.find("rtkgps") != j.end()) {
            poseHistories[api::PoseHistory::RTK_GPS].push_back(readGps(j, "rtkgps"));
        }
        else if (j.find("gnssEnu") != j.end()) {
            const auto &mean = j["gnssEnu"]["mean"];
            poseHistories[api::PoseHistory::RTK_GPS].push_back(api::Pose {
                .time = j["time"].get<double>(),
                .position = api::Vector3d {
                    mean[0].get<double>(),
                    mean[1].get<double>(),
                    mean[2].get<double>(),
                },
                // TODO: read gnssEuler
                .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 }
            });
        }
    }

    return poseHistories;
}

void InputJSONL::setParameters(CommandLineParameters &cmdParameters) {
    if (parametersPath.empty()) return;
    std::ifstream parametersFile(parametersPath);
    assert(parametersFile.is_open());
    if (isYamlFile(parametersPath)) {
        cmdParameters.parse_yaml_config(parametersFile);
    } else {
        cmdParameters.parse_algorithm_parameters(parametersFile);
    }
}

std::string InputJSONL::getParametersString() const {
    return getParametersStringWithPath(parametersPath);
}

bool InputJSONL::getParametersAvailable() const {
    return parametersAvailable;
}

std::string InputJSONL::getLastJSONL() const {
    return line;
}

bool InputJSONL::canEcho() const {
    return true;
}

api::Pose InputJSONL::readGps(const json &j, const std::string &field) {
    json jGps = j[field];
    const Eigen::Vector3d gps = gpsToLocal.convert(
        jGps["latitude"],
        jGps["longitude"],
        jGps["altitude"],
        jGps["accuracy"]);
    return api::Pose {
        .time = j["time"].get<double>(),
        .position = api::Vector3d { gps.x(), gps.y(), gps.z() },
        .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 }
    };
}


} // namespace

namespace odometry {

std::unique_ptr<InputI> buildInputJSONL(
    const std::string &inputFilePath,
    bool requireAllFiles,
    const std::string &parametersPathCustom
) {
    return std::unique_ptr<InputI>(new InputJSONL(inputFilePath, requireAllFiles, parametersPathCustom));
}

} // namespace odometry
