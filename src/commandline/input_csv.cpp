// Read sensor data and other input from multiple CSV files. Supports
// 1) a folder containing `data.csv` with video and metadata.
// 2) legacy format with `<name>.csv` and `<name>.mov` (or mp4) in same directory.

#include "input.hpp"
#include "parameters.hpp"
#include "../util/util.hpp"
#include "../util/gps.hpp"

#include <iostream>

namespace {
using namespace odometry;

const int IMU_FRAME = 1;
const int IMU_GPS = 2;
const int IMU_ACCELEROMETER = 3;
const int IMU_GYROSCOPE = 4;
const int IMU_ARKIT = 7;

class InputCSV : public InputI {
public:
    InputCSV(std::string inputFilePath);
    InputSample nextType() final;
    void getGyroscope(double& t, api::Vector3d& p) final;
    void getAccelerometer(double& t, api::Vector3d& p) final;
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
    std::string inputFolder;
    std::string inputVideoPath;
    std::string inputPath;
    std::ifstream inputFile;
    std::string parametersPath;
    std::string groundTruthPath;
    std::string previousRunPath;
    int currentCameraInd;
    int syncedFrameInd;
    double t;
    api::Vector3d p;
    int ind;
    double focalLength;
    bool parametersAvailable;
    util::GpsToLocalConverter gpsToLocal;
};

InputCSV::InputCSV(std::string inputFileNameOrPath) :
    inputVideoPath(),
    inputPath(),
    inputFile(),
    currentCameraInd(0),
    syncedFrameInd(-1),
    t(0.0),
    p(),
    ind(0),
    focalLength(0.0),
    parametersAvailable(true)
{
    // backwards compatibility: check if a video file name is given
    {
        const std::string inputPathWithoutSuffix = util::removeFileSuffix(inputFileNameOrPath);
        const std::string inputSuffix = inputFileNameOrPath.substr(inputPathWithoutSuffix.size());
        if (inputSuffix == ".mp4" || inputSuffix == ".mov") {
            inputVideoPath = inputFileNameOrPath;
            inputPath = inputPathWithoutSuffix + ".csv";
            inputFolder = util::getFileUnixPath(inputFileNameOrPath);
            parametersAvailable = false;
        }
    }

    if (inputFolder.empty()) inputFolder = util::stripTrailingSlash(inputFileNameOrPath);

    // find out video file name
    if (inputVideoPath.empty()) {
        inputVideoPath = util::joinUnixPath(inputFolder, "data.mp4");
        std::ifstream testF(inputVideoPath);
        if (testF.is_open()) {
            testF.close();
        } else {
            inputVideoPath = util::joinUnixPath(inputFolder, "data.mov");
        }
    }

    if (inputPath.empty()) {
        inputPath = util::joinUnixPath(inputFolder, "data.csv");
    }

    inputFile.open(inputPath);
    if (!inputFile.is_open()) {
        std::cerr << "Could not open " << inputPath << std::endl;
        assert(false);
    }

    // check if these exist. clear otherwise
    groundTruthPath = util::joinUnixPath(inputFolder, "ground-truth.csv");
    previousRunPath = util::joinUnixPath(inputFolder, "pose.csv");
    parametersPath = util::joinUnixPath(inputFolder, "parameters.txt");

    std::ifstream parametersFile(parametersPath);
    if (parametersFile.is_open()) {
        parametersFile.close();
    } else {
        parametersPath.clear();
    }

    std::ifstream gtFile(groundTruthPath);
    if (gtFile.is_open()) {
        gtFile.close();
    } else {
        groundTruthPath.clear();
    }

    std::ifstream previousRunFile(previousRunPath);
    if (previousRunFile.is_open()) {
        gtFile.close();
    } else {
        previousRunPath.clear();
    }
}

InputSample InputCSV::nextType() {
    if (!inputFile.is_open()) {
        std::cout << "not open" << std::endl;
        return InputSample::NONE;
    }
    std::string line;
    while (std::getline(inputFile, line)) {
        std::vector<double> v;
        std::string token;
        std::istringstream ss(line);
        while (std::getline(ss, token, ',')) {
            v.push_back(std::stod(token));
        }
        t = v[0];
        int imuType = static_cast<int>(v[1]);

        if (imuType == IMU_GYROSCOPE) {
            double x = v[2];
            double y = v[3];
            double z = v[4];
            p = { x, y, z };
            return InputSample::GYROSCOPE;
        }
        else if (imuType == IMU_ACCELEROMETER) {
            double x = v[2];
            double y = v[3];
            double z = v[4];
            p = { x, y, z };
            return InputSample::ACCELEROMETER;
        }
        else if (imuType == IMU_ARKIT) {
            ind = static_cast<int>(v[2]);
            if (v.size() >= 11) {
                const double fl = (v[9] + v[10]) / 2.0;
                if (fl > 0.0) {
                    focalLength = fl;
                }
            }
            // principal = cv::Point2d(v[11], v[12]);
            return InputSample::FRAME;
        }
        else if (imuType == IMU_FRAME) {
            ind = static_cast<int>(v[2]);
            // Older data sets don't have these fields so check number of fields.
            if (v.size() >= 7) {
                focalLength = (v[3] + v[4]) / 2.0;
                // principal = cv::Point2d(v[5], v[6]);
            }
            if (v.size() >= 8) {
                currentCameraInd = static_cast<int>(v[7]);
            }
            else {
                currentCameraInd = 0;
            }

            if (v.size() >= 9) {
                syncedFrameInd = static_cast<int>(v[8]);
            }
            else {
                syncedFrameInd = -1;
            }
            return InputSample::FRAME;
        }
        // else loop the while to get next line.
    }
    return InputSample::NONE;
}

void InputCSV::getGyroscope(double& _t, api::Vector3d& _p) {
    _t = t;
    _p = p;
}

void InputCSV::getAccelerometer(double& _t, api::Vector3d& _p) {
    _t = t;
    _p = p;
}

void InputCSV::getFrames(double &t, int &framesInd, std::vector<InputFrame> &frames) {
    t = this->t;
    framesInd = this->ind;
    frames.clear();
    frames.push_back(InputFrame {
        .intrinsic = api::CameraParameters(focalLength),
        .t = this->t
    });
}

std::string InputCSV::getInputVideoPath(int cameraInd) const {
    if (cameraInd > 0) {
        std::string fileName = "data" + std::to_string(cameraInd + 1) + ".mp4";
        return util::joinUnixPath(inputFolder, fileName);
    }
    else {
        return inputVideoPath;
    }
}

void InputCSV::setAlgorithmParametersFromData(odometry::Parameters &parameters) {
    (void)parameters;
}

std::map<api::PoseHistory, std::vector<api::Pose>>
InputCSV::getPoseHistories()
{
    std::map<api::PoseHistory, std::vector<api::Pose>> poseHistories;

    // Ground truth.
    {
        std::ifstream inputFile(groundTruthPath);
        std::string line;
        while (inputFile.is_open() && std::getline(inputFile, line)) {
            std::vector<double> v;
            std::string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ',')) v.push_back(std::stod(token));
            poseHistories[api::PoseHistory::GROUND_TRUTH].push_back(api::Pose {
                .time = v[0],
                .position = { .x = v[3], .y = v[1], .z = v[2] },
                // TODO: orientation
                .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 },
            });
        }
    }

    // Previous run.
    {
        std::ifstream inputFile(previousRunPath);
        std::string line;
        while (inputFile.is_open() && std::getline(inputFile, line)) {
            std::vector<double> v;
            std::string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ',')) v.push_back(std::stod(token));
            poseHistories[api::PoseHistory::OUR_PREVIOUS].push_back(api::Pose {
                .time = v[0],
                .position = { .x = v[1], .y = v[2], .z = v[3] },
                // TODO: correct?
                .orientation = { .x = v[4], .y = v[5], .z = v[6], .w = v[7] },
            });
        }
    }

    // ARKit and GPS (Apple location service).
    {
        std::ifstream inputFile(inputPath);
        std::string line;
        while (inputFile.is_open() && std::getline(inputFile, line)) {
            std::vector<double> v;
            std::string token;
            std::istringstream ss(line);
            while (std::getline(ss, token, ',')) {
                v.push_back(std::stod(token));
            }
            int imuType = static_cast<int>(v[1]);

            if (imuType == IMU_ARKIT) {
                poseHistories[api::PoseHistory::ARKIT].push_back(api::Pose {
                    .time = v[0],
                    .position = { .x = v[5], .y = v[3], .z = v[4] },
                    // ARKit outputs rotation matrix but not quaternion.
                    .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 },
                });
            }
            else if (imuType == IMU_GPS) {
                const Eigen::Vector3d gps = gpsToLocal.convert(v[2], v[3], v[5], v[4]);
                poseHistories[api::PoseHistory::GPS].push_back(api::Pose {
                    .time = v[0],
                    .position = api::Vector3d { gps.x(), gps.y(), gps.z() },
                    .orientation = { .x = 0, .y = 0, .z = 0, .w = 0 },
                });
            }
        }
    }

    return poseHistories;
}

void InputCSV::setParameters(CommandLineParameters &parameters) {
    if (parametersPath.empty()) return;
    std::ifstream paramFile(parametersPath);
    assert(paramFile.is_open());
    parameters.parse_algorithm_parameters(paramFile);
}

std::string InputCSV::getParametersString() const {
    return getParametersStringWithPath(parametersPath);
}

bool InputCSV::getParametersAvailable() const {
    return parametersAvailable;
}

std::string InputCSV::getLastJSONL() const {
    // CSV input should never return ECHO_RECORDING, so this should not be called by main.
    assert(false);
    return "";
}

bool InputCSV::canEcho() const {
    return false;
}

} // namespace

namespace odometry {

std::unique_ptr<InputI> buildInputCSV(const std::string &inputFilePath) {
    return std::unique_ptr<InputI>(new InputCSV(inputFilePath));
}

} // namespace odometry
