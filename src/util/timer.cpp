#include "timer.hpp"

#include <iomanip>
#include <sstream>

// Global variables.
namespace odometry {
    std::unique_ptr<util::TimeStats> TIME_STATS = nullptr;
}
namespace slam {
    std::unique_ptr<util::TimeStats> TIME_STATS = nullptr;
}

namespace util {

constexpr size_t TIMER_TEXT_WIDTH = 30;

TimeStats::Timer::Timer(TimeStats &stats, const char *name) :
    t(std::chrono::steady_clock::now()), name(std::string(name)), stats(stats) {}

TimeStats::Timer::~Timer() {
    const auto time
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
                    .count();
    stats.addTime(name, time);
}

TimeStats::Timer TimeStats::time(const char *name) {
    std::lock_guard<std::mutex> l(m);
    return Timer(*this, name);
}

void TimeStats::addTime(const std::string &name, double time) {
    std::lock_guard<std::mutex> l(m);
    timings[name] = time;
    if (!accumulatedTimings.count(name)) {
        accumulatedTimings[name] = { 0.0, 0.0 };
    }
    accumulatedTimings[name].first += time;
    accumulatedTimings[name].second += 1.0;
}

std::string TimeStats::previousTimings() {
    std::lock_guard<std::mutex> l(m);
    double total = 0.0;
    std::stringstream ss;
    ss << "Previous timings [ms]:";
    for (const auto &p : timings) {
        total += p.second;
        ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << p.first;
        ss << std::setw(0) << "   " << p.second;
    }
    ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << "TOTAL";
    ss << std::setw(0) << "   " << total;
    return ss.str();
}

std::string TimeStats::averageTimings() {
    std::lock_guard<std::mutex> l(m);
    double total = 0.0;
    std::stringstream ss;
    ss << "Per-call average timings [ms]:";
    for (const auto &p : accumulatedTimings) {
        if (p.second.second > 0.0) {
            double m = p.second.first / p.second.second;
            total += m;
            ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << p.first;
            ss << std::setw(0) << "   " << m << " = " << p.second.first << " / " << p.second.second;
        }
    }
    ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << "TOTAL";
    ss << std::setw(0) << "   " << total;
    return ss.str();
}

std::string TimeStats::perFrameTimings() {
    std::lock_guard<std::mutex> l(m);
    double total = 0.0;
    std::stringstream ss;
    ss << "Per-frame average timings [ms]:";
    if (frameCount > 0.0) {
        for (const auto &p : accumulatedTimings) {
            double m = p.second.first / frameCount;
            total += m;
            ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << p.first;
            ss << std::setw(0) << "   " << m;
        }
    }
    ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << "TOTAL";
    ss << std::setw(0) << "   " << total;
    ss << "\n" << std::setw(TIMER_TEXT_WIDTH) << "FRAMES";
    ss << std::setw(0) << "   " << frameCount;
    return ss.str();
}

void TimeStats::startFrame() {
    std::lock_guard<std::mutex> l(m);
    frameCount += 1.0;
}

std::unique_ptr<TimeStats::Timer> createTimer(const std::unique_ptr<TimeStats> &timeStats, const char *name) {
    if (!timeStats) return nullptr;
    return std::make_unique<TimeStats::Timer>(timeStats->time(name));
}

} // namespace util
