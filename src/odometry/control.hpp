#ifndef ODOMETRY_CONTROL_H_
#define ODOMETRY_CONTROL_H_

#include <functional>
#include <memory>
#include <Eigen/Dense>

#include "sample_sync.hpp"
#include "../tracker/track.hpp"
#include "output.hpp"
#include "../api/slam_map_point_record.hpp"

namespace odometry { class DebugAPI; }

namespace odometry {
class EKF;
struct Parameters;

struct BackEndBase {
    virtual ~BackEndBase();

    virtual void lockBiases() = 0;
    virtual void conditionOnLastPose() = 0;

    // for internal APIs, use with caution
    virtual const EKF &getEKF() const = 0;

    virtual void connectDebugAPI(odometry::DebugAPI &odometryDebug) = 0;
    virtual std::string stateAsString() const = 0; // would be nicer if this used DebugAPI somehow
};

struct BackEnd : BackEndBase {
    static std::unique_ptr<BackEnd> build(const Parameters &parameters);
    // Makes implementing "reset" slightly easier
    static std::unique_ptr<BackEnd> build(std::unique_ptr<BackEnd> previous);

    enum class ProcessResult {
        NONE,
        FRAME,
        SLAM_FRAME
    };
    virtual ProcessResult process(SyncedSample &sample, Output &output) = 0;

    virtual void initializeAtPose(const Eigen::Vector3d &pos, const Eigen::Vector4d &q) = 0;
};

// Control logic for an odometry session.
class Control : public BackEndBase {
protected:
    using ImagePtr = std::unique_ptr<tracker::Image>;

public:
    static std::unique_ptr<Control> build(const Parameters &parameters);
    virtual ~Control() = default;

    // reset algorithm state
    virtual void reset(bool keepPose) = 0;

    virtual void processGyroSample(double t, const api::Vector3d &p) = 0;
    virtual void processAccelerometerSample(double t, const api::Vector3d &p) = 0;

    enum class SampleProcessResult {
        NONE,
        SYNCED_SAMPLES,
        FRAMES
    };

    virtual SampleProcessResult processSyncedSamples(int maxCount) = 0;

    virtual void processFrame(
        double t,
        ImagePtr grayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) = 0;
    virtual void processStereoFrames(
        double t,
        ImagePtr firstGrayFrame,
        ImagePtr secondGrayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) = 0;

    virtual Output getOutput() const = 0;
};

} // namespace odometry

#endif // ODOMETRY_CONTROL_H_
