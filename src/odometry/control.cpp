#include "control.hpp"
#include "debug.hpp"
#include "ekf.hpp"
#include "tagged_frame.hpp"
#include "util.hpp"
#include "../tracker/camera.hpp"
#include "../tracker/image.hpp"
#include "../util/logging.hpp"
#include "../util/timer.hpp"

#include <Eigen/Dense>

namespace odometry {
namespace {

/**
 * Glue. Holds some objects we do not want or need to reset even if we
 * internally reset the algorithm.
 */
class ControlImplementation : public Control {
private:
    const Parameters &parameters;

    // Current tracking session. Empty if uninitialized or not tracking
    std::unique_ptr<BackEnd> session;

    std::unique_ptr<SampleSync> sampleSync;
    Output output;

    api::TrackingStatus controlTrackingStatus;
    double lastResetTime;
    double imuToCameraTimeShiftThresholdSeconds;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ControlImplementation(const Parameters &parameters_)
    :
        parameters(parameters_),
        session(),
        sampleSync(SampleSync::build(parameters)),
        controlTrackingStatus(api::TrackingStatus::INIT),
        lastResetTime(0.0),
        imuToCameraTimeShiftThresholdSeconds(0.01)
    {
        reset();
    }

    void reset(bool keepPose = false) final {
        if (session) {
            lastResetTime = session->getEKF().getPlatformTime();
        }
        if (keepPose) {
            assert(session);
            // NOTE: never use auto with Eigen types!
            const Eigen::Vector3d pos = session->getEKF().position();
            const Eigen::Vector4d q = session->getEKF().orientation();
            if (session) session = BackEnd::build(std::move(session));
            else session = BackEnd::build(parameters);
            session->initializeAtPose(pos, q);
        } else {
            if (session) session = BackEnd::build(std::move(session));
            else session = BackEnd::build(parameters);
        }
    }

    void processGyroSample(double t, const api::Vector3d &p) final {
        sampleSync->addSampleLeader(t, p);
    }

    void processAccelerometerSample(double t, const api::Vector3d &p) final {
        sampleSync->addSampleFollower(t, p);
    }

    /**
     * @param maxCount If > 0, process up to that many synced samples.
     * @return summary of what was processed
     */
    SampleProcessResult processSyncedSamples(int maxCount) final {
        const ParametersOdometry &po = parameters.odometry;

        int processedSamples = 0, processedFrames = 0;

        SyncedSample syncedSample;
        Output tmpOutput;
        while (sampleSync->pollSyncedSample(syncedSample)) {
            // only process data from sample sync if session is active,
            // i.e., now in tracking state
            if (session) {
                if (syncedSample.frame && odometry::TIME_STATS) {
                    odometry::TIME_STATS->startFrame();
                }
                const auto beResult = session->process(syncedSample, tmpOutput);
                if (beResult != BackEnd::ProcessResult::NONE) {
                    processedFrames++;
                }
                if (parameters.odometry.estimateImuCameraTimeShift) {
                    double shift = session->getEKF().getImuToCameraTimeShift();
                    sampleSync->setImuToCameraTimeShift(shift);
                    if (std::abs(shift) > imuToCameraTimeShiftThresholdSeconds) {
                        // The odometry may well work above the threshold, but warn in case
                        // the variable could "explode" into non-sensical values.
                        log_warn("Large imu-to-camera time shift %.3fs.", shift);
                        imuToCameraTimeShiftThresholdSeconds *= 2.0;
                    }
                }
            }

            processedSamples++;
            if (maxCount > 0 && processedSamples >= maxCount) break;
        }

        // NOTE: this function may be called at ~400 Hz so avoid any
        // non-essential processing in the base case (nothing "moved")

        // Update output if any frames were processed (incl. non-keyframes)
        if (processedFrames > 0) {
            const double t = session->getEKF().getPlatformTime();
            tmpOutput.t = t; // TODO: check this

            const auto sessionTrackingStatus = tmpOutput.trackingStatus;
            tmpOutput.trackingStatus = controlTrackingStatus;

            // Run all the above and discard the results if freezed, so that jump filter still keeps running.
            const bool frozen = po.freezeOnFailedTracking
                && controlTrackingStatus != api::TrackingStatus::INIT
                && sessionTrackingStatus != api::TrackingStatus::TRACKING;
            if (!frozen) output = tmpOutput;

            // Update persistent tracking status, but prevent it going back to INIT
            if (controlTrackingStatus == api::TrackingStatus::INIT || sessionTrackingStatus != api::TrackingStatus::INIT) {
                controlTrackingStatus = sessionTrackingStatus;
            }

            // Trigger reset when tracking fails
            bool resetTimerExpired = lastResetTime + po.resetAfterTrackingFailsToInitialize < t;
            // TODO: Could be used to check odometry status, but couldn't find good way to incorporate it to the logic here
            // bool wasStationary = session->getEKF().getWasStationary();
            if (controlTrackingStatus == api::TrackingStatus::INIT && resetTimerExpired && po.resetUntilInitSucceeds) {
                log_debug("First time init failed to track in %f seconds, reseting", po.resetAfterTrackingFailsToInitialize);
                reset(false);
            } else if (po.resetOnFailedTracking && sessionTrackingStatus == api::TrackingStatus::LOST_TRACKING) {
                log_debug("Lost tracking, reseting");
                reset(true);
            } else if (controlTrackingStatus != api::TrackingStatus::INIT && sessionTrackingStatus == api::TrackingStatus::INIT
                && resetTimerExpired) {
                log_debug("Failed to initialize tracking in %f seconds, reseting", po.resetAfterTrackingFailsToInitialize);
                reset(true);
            }
        }

        if (processedFrames > 0) return SampleProcessResult::FRAMES;
        if (processedSamples > 0) return SampleProcessResult::SYNCED_SAMPLES;
        return SampleProcessResult::NONE;
    }

    void processFrame(
        double t,
        ImagePtr grayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) final {
        processStereoFrames(t, std::move(grayFrame), {}, std::move(taggedFrame));
    }

    void processStereoFrames(
        double t,
        ImagePtr firstGrayFrame,
        ImagePtr secondGrayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) {
        // Store the frames as they will be used only after sample sync produces a sample.
        sampleSync->addFrame(
            t,
            std::move(firstGrayFrame),
            std::move(secondGrayFrame),
            std::move(taggedFrame)
        );
    }

    void lockBiases() final {
        assert(session);
        session->lockBiases();
    }

    void conditionOnLastPose() final {
        assert(session);
        session->conditionOnLastPose();
    }

    Output getOutput() const final {
        return output;
    }

    const EKF &getEKF() const final {
        assert(session);
        return session->getEKF();
    }

    void connectDebugAPI(odometry::DebugAPI &odometryDebug) final {
        assert(session); // TODO
        session->connectDebugAPI(odometryDebug);
    }

    std::string stateAsString() const final {
        if (!session) return "";
        return session->stateAsString();
    }
};
} // anonymous namespace

std::unique_ptr<Control> Control::build(const Parameters &p) {
    return std::unique_ptr<Control>(new ControlImplementation(p));
}
} // namespace odometry
