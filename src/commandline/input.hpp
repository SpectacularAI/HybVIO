#ifndef ODOMETRY_INPUT_H_
#define ODOMETRY_INPUT_H_

#include <fstream>

#include <Eigen/Dense>

#include "../api/vio.hpp"
#include "../api/internal.hpp"
#include "../tracker/camera.hpp"

struct CommandLineParameters; // fwd decl
namespace odometry {

struct InputFrame {
    api::CameraParameters intrinsic;
    double t;
};

enum class InputSample {
    NONE = 0,
    GYROSCOPE = 1,
    ACCELEROMETER = 2,
    FRAME = 3,
    ECHO_RECORDING = 4
};

// Usage: In a loop, call nextType() and then the getX() function hinted by
// the return type.
class InputI {
public:
    virtual ~InputI() {};
    virtual InputSample nextType() = 0;
    virtual void getGyroscope(double& t, api::Vector3d& p) = 0;
    virtual void getAccelerometer(double& t, api::Vector3d& p) = 0;
    virtual void getFrames(double &t, int &framesInd, std::vector<InputFrame> &frames) = 0;
    virtual std::string getInputVideoPath(int cameraInd) const = 0;
    virtual void setAlgorithmParametersFromData(odometry::Parameters &parameters) = 0;

    /**
     * Read algorithm parameters.
     * @param cmdParameters Struct to which parameters will be placed.
     */
    virtual void setParameters(CommandLineParameters &cmdParameters) = 0;
    virtual std::string getParametersString() const = 0;
    virtual bool getParametersAvailable() const = 0;
    virtual std::map<api::PoseHistory, std::vector<api::Pose>> getPoseHistories() = 0;
    virtual std::string getLastJSONL() const = 0;
    /**
     * True for JSONL input, false otherwise.
     */
    virtual bool canEcho() const = 0;
};

std::unique_ptr<InputI> buildInputCSV(const std::string &inputFilePath);
std::unique_ptr<InputI> buildInputJSONL(
    const std::string &inputFilePath,
    bool requireAllFiles = true,
    const std::string &parametersPathCustom = ""
);

// Helpers shared by the CSV and JSONL classes.
bool pathHasFile(const std::string &path, const std::string &file);
std::string getParametersStringWithPath(const std::string &parametersPath);

} // namespace odometry

#endif // ODOMETRY_INPUT_H_
