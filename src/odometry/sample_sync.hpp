// I would feel more confident in the implementation if there was also some kind of visualization
// for the sample sync algorithm. For example it could be an animation which shows gyro, acc and
// frame samples on three different timelines and shows which get paired together and how the
// pairs update (frames are paired in a dynamical manner).

#ifndef DAZZLING_SAMPLE_SYNC_H_
#define DAZZLING_SAMPLE_SYNC_H_

#include <memory>
#include "../odometry/parameters.hpp"
#include "../api/types.hpp"

namespace tracker { struct Image; }
namespace odometry {

using ImagePtr = std::unique_ptr<tracker::Image>;
struct TaggedFrame;
struct ProcessedFrame;

struct Sample {
    double t;
    api::Vector3d p;
};

struct SyncedSample {
    // Timestamp of leader sample. Use this in algorithms.
    double t;
    // Timestamp of follower sample. Can compare this to `t` to detect problems.
    double tF;
    // Leader sample.
    api::Vector3d l;
    // Follower sample.
    api::Vector3d f;
    // Optional camera frame.
    std::unique_ptr<ProcessedFrame> frame;

    SyncedSample();
    ~SyncedSample();
};

// A sensor sample synchronizer capable of handling time shift
// between sensor outputs and incorrectly ordered samples.
// One of the IMU sensors is assigned as a "leader" (L in variables)
// on whose clock the synchronized samples are given (likely gyroscope).
// Other IMU sensors are labeled as "followers" (F, code supports just one
// for now, likely the accelerometer). The synchronizer indicates Ready
// status when it can output a set of of synchronized samples.
// The class instances are not thread-safe (so use only one sensor
// processing thread/queue).
// The `lag` parameter is roughly the number of samples the output
// gets delayed for the leader sensor.
class SampleSync {
public:
    static std::unique_ptr<SampleSync> build(const Parameters& p);
    virtual ~SampleSync();

    virtual void addFrame(
        double t,
        ImagePtr firstGrayFrame,
        ImagePtr secondGrayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) = 0;
    virtual void addSampleFollower(double t, const api::Vector3d& p) = 0;
    virtual void addSampleLeader(double t, const api::Vector3d& p) = 0;
    virtual bool pollSyncedSample(SyncedSample &out) = 0;
    virtual void setImuToCameraTimeShift(double t) = 0;
};

} // namespace odometry

#endif // DAZZLING_SAMPLE_SYNC_H_
