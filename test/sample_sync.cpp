#include "catch2/catch.hpp"
#include <opencv2/opencv.hpp>

#include "../src/odometry/processed_frame.hpp"
#include "../src/odometry/tagged_frame.hpp"
#include "../src/odometry/sample_sync.hpp"
#include "../src/tracker/image.hpp"

namespace {
// hacky fix for the fact that parameters are used by reference in SampleSync
std::unique_ptr<odometry::Parameters> parametersGlobal;
}

std::unique_ptr<odometry::SampleSync> makeSampleSync() {
    if (!parametersGlobal) {
        parametersGlobal = std::make_unique<odometry::Parameters>();
        parametersGlobal->odometry.sampleSyncLag = 25;
        parametersGlobal->odometry.visualUpdateEnabled = true;
    }
    return odometry::SampleSync::build(*parametersGlobal);
}

// Check nothing crashes and no asserts trigger if the synced
// samples are not consumed.
TEST_CASE("noChoke", "[sampleSync]") {
    auto ss = makeSampleSync();
    double t = 5.0;
    while (t < 8.0) {
        ss->addSampleLeader(t, { t, t, t });
        ss->addSampleFollower(t, { t, t, t });
        t += 0.01;
    }
}

// Test syncing with frames.
TEST_CASE("frameTest", "[sampleSync]") {
    auto ss = makeSampleSync();
    // Simulate roughly 100Hz sensor for easier intuition.
    const double dt = 0.01;
    const double camlag = 0.002;

    std::vector<size_t> outputFrameNums;

    double t = 1.0;
    int i = 0;
    while (t < 5.0) {
        ss->addSampleLeader(t, { t, t, t });
        ss->addSampleFollower(t, { t, t, t });
        if (i % 10 == 3) {
            ss->addFrame(t + camlag, {}, {}, nullptr);
        }

        // Drain all available samples.
        odometry::SyncedSample sample;
        while (ss->pollSyncedSample(sample)) {
            if (sample.frame != nullptr) {
                size_t num = sample.frame->num;
                // Check we get frame output 1, 2, 3, ...
                if (outputFrameNums.size() > 0) {
                    REQUIRE(num == outputFrameNums[outputFrameNums.size() - 1] + 1);
                }
                else {
                    REQUIRE(num == 1);
                }
                // Check the leaderTimeDiff value is correct.
                REQUIRE(std::abs(sample.frame->leaderTimeDiff - camlag) < 0.0001);

                outputFrameNums.push_back(num);
            }
        }

        i++;
        t += dt;
    }

    REQUIRE(outputFrameNums.size() > 0);
    // There may still be frames left in the buffer, but not too many.
    // The bound depends on sampling frequencies and the various buffer sizes.
    // REQUIRE(ss->frames.size() <= 2);
}

// Test various syncing capabilities.
TEST_CASE("integrationTest", "[sampleSync]") {
    auto ss = makeSampleSync();
    std::unique_ptr<odometry::SyncedSample> sample(new odometry::SyncedSample);
    REQUIRE(!ss->pollSyncedSample(*sample));

    // Simulate roughly 100Hz sensor for easier intuition.
    const double lfShift = 0.003;

    const double tAccStart = 5.1;
    const double tAccEnd = 7.8;

    // Hide a log message from sample sync about too large time differences.
    const double scale = 0.5;

    std::vector< std::unique_ptr<odometry::SyncedSample> > samples;

    double t = 5.0;
    int i = 0;
    while (t < 8.0) {
        if (t < 8.2) {
            REQUIRE(!ss->pollSyncedSample(*sample));
        }

        {
            // Simulate samples out of order.
            double tr = t;
            if (i % 6 == 2) {
                tr += 0.033 * scale;
            }
            if (i % 11 == 3) {
                tr -= 0.011 * scale;
            }
            ss->addSampleLeader(tr, { tr, tr, tr });
        }
        // Simulate roughly constant time shift between different sensors.
        t += lfShift;

        // Simulate sensors starting and stopping at different times.
        if (t > tAccStart && t < tAccEnd) {
            double tr = t;
            if (i % 7 == 3) {
                tr += 0.052 * scale;
            }
            if (i % 3 == 2) {
                tr -= 0.031 * scale;
            }
            ss->addSampleFollower(tr, { tr, tr, tr });
        }

        // Add some frames too because otherwise there is no output.
        if (i % 10 == 3) {
            ss->addFrame(t, {}, {}, nullptr);
        }

        t += (0.01 - lfShift);
        i++;

        // Drain all available samples.
        while (ss->pollSyncedSample(*sample)) {
            // Check samples come out as they were put in.
            REQUIRE(sample->t == sample->l.x);
            REQUIRE(sample->l.x == sample->l.y);
            REQUIRE(sample->f.x == sample->f.y);

            samples.push_back(std::move(sample));
            sample.reset(new odometry::SyncedSample);
        }
    }

    REQUIRE(samples.size() > 0);
    // Test order of output samples. Note that success varies on
    // the test parameters. For example if the synchronizer
    // buffers are too small the algorithm will be unable to
    // do the ordering perfectly.
    for (size_t i = 1; i < samples.size(); i++) {
        REQUIRE(samples[i]->t >= samples[i - 1]->t);
        REQUIRE(samples[i]->f.x >= samples[i - 1]->f.x);
    }
    // Test the leader and follower timestamps are close.
    for (size_t i = 0; i < samples.size(); i++) {
        if (samples[i]->t < tAccStart || samples[i]->t > tAccEnd) continue;
        REQUIRE(std::abs(samples[i]->t - samples[i]->tF) < 0.03);
    }

    // Calling getSample repeatedly will eventually make the ss not ready.
    REQUIRE(!ss->pollSyncedSample(*sample));
}
