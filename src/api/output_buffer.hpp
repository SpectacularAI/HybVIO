#ifndef DAZZLING_OUTPUT_BUFFER
#define DAZZLING_OUTPUT_BUFFER

#include "internal.hpp"
#include <deque>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <string>

namespace api {

class OutputBuffer {
public:

    struct Output {
        std::shared_ptr<const api::VioApi::VioOutput> output;
        std::string status;
    };

private:
    bool first = true;
    std::deque<std::shared_ptr<const Output>> buf;
    std::mutex mutex;

    struct Stats {
        static constexpr double UPDATE_INTERVAL_S = 1.0;
        double lastUpdateT = 0;
        int nProcessedFrames = 0, nOutputFrames = 0;
        int nSkips = 0;
        double totalDelta = 0.0, minDelta = -1, maxDelta = -1;

        std::string toString(double dt) const {
            if (dt <= 0) return "";

            std::ostringstream oss;
            oss << std::setprecision(3)
                << "FPS out: " << (nOutputFrames / dt)
                << " latency " << (totalDelta / nOutputFrames * 1000)
                << " +- " << ((maxDelta - minDelta)*1000/2) << " ms"
                << " " << (nSkips / dt) << " skips/s";

            auto r = oss.str();
            // log_debug("OutputBuffer stats: %s", r.c_str());
            return r;
        }
    } stats;

public:
    double targetDelaySeconds = 0;
    std::string statsText = "";

    void addProcessedFrame(std::shared_ptr<const api::VioApi::VioOutput> algoOut) {
        std::unique_lock<std::mutex> lock(mutex);
        if (targetDelaySeconds <= 0) buf.clear();

        // log_debug("process %f (%zu buf)", algoOut->pose.time, buf.size());

        double dt = 0;

        const double tCur = algoOut->pose.time;
        if (first) {
            first = false;
            stats.lastUpdateT = tCur;
        }
        else {
            dt = tCur - stats.lastUpdateT;
        }
        stats.nProcessedFrames++;

        if (dt > Stats::UPDATE_INTERVAL_S) {
            const auto tmpStats = stats;
            stats = Stats{};
            lock.unlock();
            statsText = tmpStats.toString(dt);
            lock.lock();
            stats.lastUpdateT = tCur;
        }

        buf.push_back(std::make_shared<Output>(Output {
            .output = algoOut,
            .status = statsText
        }));
    }

    std::shared_ptr<const Output> pollOutput(double timestamp) {
        std::lock_guard<std::mutex> lock(mutex);

        int nOut = 0;
        std::shared_ptr<const Output> out;
        // if t < 0, return all output
        while (!buf.empty() && (timestamp - buf.front()->output->pose.time >= targetDelaySeconds || timestamp < 0)) {
            nOut++;
            out = buf.front();
            buf.pop_front();
        }

        if (nOut >= 1) {
            stats.nOutputFrames++;
            stats.nSkips += nOut - 1;
            const double delta = timestamp - out->output->pose.time;
            stats.totalDelta += delta;
            if (stats.minDelta > delta || stats.minDelta < 0) stats.minDelta = delta;
            if (stats.maxDelta < delta || stats.maxDelta < 0) stats.maxDelta = delta;
            // log_debug("poll out %f, delta %g/%g", out->pose.time, delta, targetDelaySeconds);
        }
        return out;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        buf.clear();
    }
};
}

#endif
