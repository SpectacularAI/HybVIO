#ifndef UTIL_TIMER_H_
#define UTIL_TIMER_H_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace util {

/**
 *  RAII timer
 */
class TimeStats {
private:
    std::unordered_map<std::string, double> timings;
    std::unordered_map<std::string, std::pair<double, double>> accumulatedTimings;
    double frameCount = 0.0;
    std::mutex m;

public:
    class Timer {
    public:
        Timer(TimeStats &stats, const char *name);
        ~Timer();

    private:
        std::chrono::steady_clock::time_point t;
        std::string name;
        TimeStats &stats;
    };

    TimeStats() {}

    /**
     *  Create a RAII #Timer object. The object starts a timer and once it goes out
     *  of scope, stores the timing under a specified key.
     *
     *  @param name Key under which the timing will be stored.
     */
    Timer time(const char *name);

    void addTime(const std::string &name, double time);
    std::string previousTimings();
    std::string averageTimings();
    std::string perFrameTimings();
    void startFrame();
};

std::unique_ptr<TimeStats::Timer> createTimer(const std::unique_ptr<TimeStats> &timeStats, const char *name);

// The timer only works if you assign the return value to a variable.
#define timer(timeStats, name) auto _timer = createTimer(timeStats, name)

} // namespace util

// Global variables.
namespace odometry {
    extern std::unique_ptr<::util::TimeStats> TIME_STATS;
}
namespace slam {
    extern std::unique_ptr<::util::TimeStats> TIME_STATS;
}

#endif // UTIL_TIMER_H_
