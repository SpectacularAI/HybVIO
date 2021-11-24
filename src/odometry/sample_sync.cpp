#include "sample_sync.hpp"

#include "../util/logging.hpp"
#include "../tracker/image.hpp"
#include "processed_frame.hpp"
#include "tagged_frame.hpp"
#include "util.hpp"

#include <vector>
#include <cmath>
#include <mutex>

namespace {

// Increasing this improves capability of SampleSync by using more memory
// for the buffers but not increasing time lag.
const int LEADER_FILL_RATIO = 5;

template <class T> void cullBuffer(std::vector<T> &buf) {
    auto itr = buf.begin();
    int i = 0;
    constexpr int KEEP_EVERY_N = 2;
    while (itr != buf.end()) {
        if (i++ % KEEP_EVERY_N == 0) itr++;
        else itr = buf.erase(itr);
    }
}
}

namespace odometry {
SyncedSample::SyncedSample() = default;
SyncedSample::~SyncedSample() = default;

// needed to fwd declare tracker::Camera
ProcessedFrame::~ProcessedFrame() = default;
ProcessedFrame::ProcessedFrame(
    double t,
    ImagePtr firstGrayFrameUnique,
    ImagePtr secondGrayFrameUnique,
    std::unique_ptr<TaggedFrame> taggedFrameMoved
) :
    t(t),
    leaderIndex(0),
    leaderTimeDiff(-1.0), num(0),
    firstGrayFrame(std::move(firstGrayFrameUnique)),
    secondGrayFrame(std::move(secondGrayFrameUnique)),
    taggedFrame(std::move(taggedFrameMoved))
{
    if (taggedFrame) {
        taggedFrame->firstGrayFrame = firstGrayFrame;
        taggedFrame->secondGrayFrame = secondGrayFrame;
    }
}

class SampleSyncImplmentation : public SampleSync {
private:
    std::vector<std::unique_ptr<ProcessedFrame>> frames;
    size_t frameCount;

    // Sample buffers.
    std::vector<Sample> sF;
    std::vector<Sample> sL;
    // Valid samples in the buffers.
    std::vector<bool> availableL;
    size_t countF;
    size_t countL;
    // Indices of the next samples to be added.
    size_t indexF;
    size_t indexL;
    const ParametersOdometry &parameters;
    double variableImuToCameraShift;

    std::mutex mutex;

    util::ThroughputCounter inputThroughput;
    util::ThroughputCounter outputThroughput;

public:

    SampleSyncImplmentation(const Parameters& p)
    : frames(), frameCount(0), sF(), sL(),
      availableL(), countF(0), countL(0), indexF(0), indexL(0),
      parameters(p.odometry), variableImuToCameraShift(0.0)
    {
        // Adding the constant prevents the "sample being overwritten"
        // warning on very small values of `sampleSyncLag`.
        int size = 100 + LEADER_FILL_RATIO * p.odometry.sampleSyncLag;

        // Negative time is used to indicate absence of sample.
        Sample s0 = {
            .t = -1.0,
            .p = { 0, 0, 0 },
        };
        sF.assign(size, s0);
        sL.assign(size, s0);
        availableL.assign(size, false);
        assert(size == static_cast<int>(sF.size()));
        assert(size == static_cast<int>(sL.size()));
        assert(size == static_cast<int>(availableL.size()));
    }

    // Returns true when new output should be requested.
    bool isReady() {
        // Keep `lag - 1` number of leader samples in the buffer to have some time for
        // trying to reorder and sync leader and follower samples and also camera frames.
        // Small value of `lag` makes the algorithm more real-time, but also more
        // vulnerable to poor quality sensor data.
        return (!parameters.visualUpdateEnabled || frames.size() >= parameters.sampleSyncFrameCount)
            && countL >= parameters.sampleSyncLag
            && countF > 0;
    }

    // Call on every processed frame.
    void addFrame(
        double t,
        ImagePtr firstGrayFrame,
        ImagePtr secondGrayFrame,
        std::unique_ptr<TaggedFrame> taggedFrame
    ) final {
        std::lock_guard<std::mutex> lock(mutex);
        // Negate to convert from camera to IMU time.
        t -= parameters.imuToCameraShiftSeconds;
        t -= variableImuToCameraShift;

        if (frames.size() >= parameters.sampleSyncFrameBufferSize) {
            // This is heavy measure and smart frame dropping should make this a rare edge case
            log_warn("SampleSync frame buffer size %zu, culling buffer\n", frames.size());
            cullBuffer(frames);
        }

        auto frame = std::make_unique<ProcessedFrame>(
            t,
            std::move(firstGrayFrame),
            std::move(secondGrayFrame),
            std::move(taggedFrame)
        );

        frame->num = ++frameCount;

        if (parameters.sampleSyncSmartFrameRateLimiter) {
            inputThroughput.put(t);
            constexpr size_t FRAME_DROP_THRESHOLD = 2;
            if (frames.size() > FRAME_DROP_THRESHOLD) {
                float itp = inputThroughput.throughputPerSecond();
                float otp = outputThroughput.throughputPerSecond();
                if (itp > 0.0 && otp > 0.0) {
                    float framesToDropPercentage = 1. - otp / itp;
                    // Add small overhead room to eagerly drop frames before buffer full
                    framesToDropPercentage *= 1.1;
                    if (framesToDropPercentage > 0.) {
                        // TODO: n is always >= 2 so this cannot drop more than 50% of the frames
                        int n = int(std::ceil(1. / framesToDropPercentage));
                        if (frame->num % n == 0) {
                            // log_debug("Dropping frame itp %f, otp %f, framesToDropPercentage %f, n %d, num %u",
                            //     itp, otp, framesToDropPercentage, n, frame->num);
                            // Remove oldest frame, don't increment throughput
                            frames.pop_back();
                        }
                    }
                }
            }
        }

        bool first = true;
        size_t index = 0;
        double dt = -1.0;
        // Find the leader sample the frame is closest to.
        for (size_t i = 0; i < sL.size(); i++) {
            if (!availableL[i]) continue;
            double dti = std::abs(sL[i].t - frame->t);
            if (first || dti < dt) {
                index = i;
                dt = dti;
                first = false;
            }
        }
        if (first) {
            // It's normal for this to occur a few times during startup of the iOS app.
            // The frame cleanup code is simpler because we can assume
            // every frame has a valid leader sample associated to it
            // from the start.
            // log_debug("Discarding camera frame #%zu received before any leader samples.", frame->num);
            return;
        }
        if (frames.size() > 0 && frames.back()->t == t) {
            // Skip frames with identical timestamp as the previous
            // log_debug("Skip duplicate frame.", frame->num);
            return;
        }

        assert(dt >= 0);
        frame->leaderIndex = index;
        frame->leaderTimeDiff = dt;

        frames.push_back(std::move(frame));
    }

    // Call on every follower sample (likely accelerometer).
    void addSampleFollower(double t, const api::Vector3d& p) final {
        std::lock_guard<std::mutex> lock(mutex);
        if (countF < sF.size()) countF++;

        sF[indexF] = Sample {
            .t = t,
            .p = p,
        };

        indexF++;
        indexF = indexF % sF.size();
    }

    // Call on every leader sample (likely gyroscope).
    void addSampleLeader(double t, const api::Vector3d& p) final {
        std::lock_guard<std::mutex> lock(mutex);
        if (countL < sL.size()) {
            countL++;
        }
        else {
            // Handle existing sample in the index.
            assert(availableL[indexL]);
            for (size_t i1 = frames.size(); i1 > 0; i1--) {
                size_t i = i1 - 1;
                if (frames[i]->leaderIndex == indexL) {
                    // Should not happen normally. If it does, try:
                    // * Increase the LEADER_FILL_RATIO constant (uses more memory).
                    // * Increase the lag parameter given to SampleSync constructor (increases time lag).
                    // * Increase camera fps (10Hz is maybe reasonable for our tracker/odometry algorithms as of writing).
                    // * Decrease IMU sampling rates.
                    // This happens because the ratio of IMU samples to camera frames exceeds
                    // capacity of the ring buffers, see `isReady()` for more comments.
                    log_warn("Discarding camera frame #%zu due to leader sample being overwritten\n",
                                frames[i]->num);
                    // Safe to erase elements with reverse iteration.
                    frames.erase(frames.begin() + i);
                }
            }
        }

        sL[indexL] = Sample {
            .t = t,
            .p = p,
        };

        // Update leaders associated with frames.
        for (size_t i = 0; i < frames.size(); i++) {
            double dti = std::abs(t - frames[i]->t);
            if (dti < frames[i]->leaderTimeDiff) {
                frames[i]->leaderIndex = indexL;
                frames[i]->leaderTimeDiff = dti;
            }
        }

        availableL[indexL] = true;
        indexL++;
        indexL = indexL % sL.size();
    }

    // Call this once each time isReady() returns true.
    bool pollSyncedSample(SyncedSample &sample) final {
        std::lock_guard<std::mutex> lock(mutex);
        if (!isReady()) return false;

        double t = 0;
        int indexL = -1;

        // Find the oldest leader sample.
        for (size_t i = 0; i < sL.size(); i++) {
            if (availableL[i] && (indexL < 0 || sL[i].t < t)) {
                t = sL[i].t;
                indexL = static_cast<int>(i);
            }
        }
        assert(indexL >= 0);

        sample.t = sL[indexL].t;
        sample.l = sL[indexL].p;

        // Use every leader sample just once.
        sL[indexL].t = -1;
        assert(countL > 0);
        countL--;
        availableL[indexL] = false;

        double dt = -1;
        int indexF = -1;

        // Find the follower sample with closest time. Follower
        // samples can be used multiple times.
        for (size_t i = 0; i < countF; i++) {
            if (indexF < 0 || std::abs(sF[i].t - sample.t) < dt) {
                dt = std::abs(sF[i].t - sample.t);
                indexF = static_cast<int>(i);
            }
        }
        assert(indexF >= 0);
        assert(dt >= 0);

        sample.tF = sF[indexF].t;
        sample.f = sF[indexF].p;
        sample.frame.reset();

        // Pair a frame to the sample if it's the best match.
        for (size_t i1 = frames.size(); i1 > 0; i1--) {
            size_t i = i1 - 1;
            if (static_cast<int>(frames[i]->leaderIndex) == indexL) {
                if (frames[i]->leaderTimeDiff > 0.01) {
                    // If the comparison constant is suited to the sensor
                    // framerates, this print probably shouldn't happen.
                    log_warn("Camera frame #%zu with large time difference to leader: (%.4f)s",
                            frames[i]->num, frames[i]->leaderTimeDiff);
                }
                // If multiple frames have same leader index, then the loop frees
                // memory of all but the last pointer, which remains moved to the sample.
                sample.frame = std::move(frames[i]);
                // Safe to erase elements with reverse iteration.
                frames.erase(frames.begin() + i);
            }
        }

        if (parameters.sampleSyncSmartFrameRateLimiter && sample.frame) {
            // Only count successfully processed frames to get processing throughput, dropped don't count
            outputThroughput.put(sample.t);
        }

        return true;
    }

    void setImuToCameraTimeShift(double t) final {
        std::lock_guard<std::mutex> lock(mutex);
        variableImuToCameraShift = t;
    }
};

SampleSync::~SampleSync() = default;
std::unique_ptr<SampleSync> SampleSync::build(const Parameters& p) {
    return std::unique_ptr<SampleSync>(new SampleSyncImplmentation(p));
}

} // namespace odometry
