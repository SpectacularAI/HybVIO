#ifndef ODOMETRY_VISUAL_UPDATE_STATS_H_
#define ODOMETRY_VISUAL_UPDATE_STATS_H_

#include "../util/logging.hpp"

namespace odometry {
namespace {

class VisualUpdateStats {
private:
    enum class VuTrack {
        USED,
        NOT_ENOUGH_FRAMES,
        BLACKLISTED,
        BAD_TRIANGULATION,
        BAD_PREPARE,
        OUTLIER_RMSE,
        OUTLIER_CHI2,
        TRIANGULATION_FOR_POINT_CLOUD,
        UNKNOWN,
        // Marker for iteration.
        LAST
    };

    const int last = static_cast<int>(VuTrack::LAST);
    bool enabled;
    int trackCount = 0;
    int totalTrackCount = 0;

    std::map<VuTrack, int> tracks;
    std::map<VuTrack, int> totalTracks;

public:
    VisualUpdateStats(bool enabled) :
        enabled(enabled)
    {
        if (!enabled) return;
        for (int i = 0; i < last; ++i) {
            VuTrack t = static_cast<VuTrack>(i);
            tracks.emplace(t, 0);
            totalTracks.emplace(t, 0);
        }
    }

    void newTrack() {
        if (!enabled) return;
        ++trackCount;
    }

    void notEnoughFrames() {
        if (!enabled) return;
        ++tracks.at(VuTrack::NOT_ENOUGH_FRAMES);
    }

    void blacklisted() {
        if (!enabled) return;
        ++tracks.at(VuTrack::BLACKLISTED);
    }

    void triangulationForPointCloud() {
        if (!enabled) return;
        ++tracks.at(VuTrack::TRIANGULATION_FOR_POINT_CLOUD);
    }

    void fullyProcessedTrack(
        TriangulatorStatus triangulateStatus,
        PrepareVuStatus prepareVuStatus,
        VuOutlierStatus outlierStatus,
        bool doVisualUpdate
    ) {
        if (!enabled) return;
        // Categorize by first failure.
        switch (triangulateStatus) {
            case TriangulatorStatus::OK:
                break;
            case TriangulatorStatus::HYBRID:
                break;
            case TriangulatorStatus::BEHIND:
            case TriangulatorStatus::BAD_COND:
            case TriangulatorStatus::NO_CONVERGENCE:
            case TriangulatorStatus::BAD_DEPTH:
                // fallthrough
            case TriangulatorStatus::UNKNOWN_PROBLEM:
                ++tracks.at(VuTrack::BAD_TRIANGULATION);
                if (doVisualUpdate) log_warn("Visual update stats logic bug? 1");
                return;
        }
        switch (prepareVuStatus) {
            case PrepareVuStatus::PREPARE_VU_OK:
                break;
            case PrepareVuStatus::PREPARE_VU_ZERO_DEPTH:
                // fallthrough
            case PrepareVuStatus::PREPARE_VU_BEHIND:
                ++tracks.at(VuTrack::BAD_PREPARE);
                if (doVisualUpdate) log_warn("Visual update stats logic bug? 2");
                return;

        }
        if (outlierStatus == VuOutlierStatus::RMSE) {
            ++tracks.at(VuTrack::OUTLIER_RMSE);
            return;
        }
        if (outlierStatus == VuOutlierStatus::CHI2) {
            ++tracks.at(VuTrack::OUTLIER_CHI2);
            return;
        }
        assert(outlierStatus == VuOutlierStatus::INLIER || outlierStatus == VuOutlierStatus::NOT_COMPUTED);
        if (doVisualUpdate) {
            ++tracks.at(VuTrack::USED);
            return;
        }
        log_warn("Visual update stats logic bug? 3");
        ++tracks.at(VuTrack::UNKNOWN);
    }

    void finishFrame() {
        if (!enabled) return;
        totalTrackCount += trackCount;
        int count = 0;
        for (int i = 0; i < last; ++i) {
            VuTrack t = static_cast<VuTrack>(i);
            totalTracks.at(t) += tracks.at(t);
            count += tracks.at(t);
        }
        // Should be zero, but won't be if the loop over tracks does `continue;`
        // without telling us about the track.
        int unknown = trackCount - count > 0 ? trackCount - count : 0;
        tracks.at(VuTrack::UNKNOWN) += unknown;
        totalTracks.at(VuTrack::UNKNOWN) += unknown;

        const char *names[10] = {
            "used              ",
            "too short         ",
            "blacklisted       ",
            "bad triangulation ",
            "bad prepare       ",
            "outlier RMSE      ",
            "outlier chi2      ",
            "triangulation only",
            "unknown           ",
            "TOTAL             ",
        };
        log_info(" ");
        log_info("TYPE                     \tNUM\tTOTAL");
        int sum = 0;
        int totalSum = 0;
        for (int i = 0; i < last; ++i) {
            VuTrack t = static_cast<VuTrack>(i);
            sum += tracks.at(t);
            totalSum += totalTracks.at(t);
            log_info("%s\t%d\t%d", names[i], tracks.at(t), totalTracks.at(t));
        }
        log_info("%s\t%d\t%d", names[last], sum, totalSum);

        trackCount = 0;
        for (int i = 0; i < last; ++i) {
            VuTrack t = static_cast<VuTrack>(i);
            tracks.at(t) = 0;
        }
    }
};

} // anonymous namespace
} // namespace odometry

#endif // ODOMETRY_VISUAL_UPDATE_STATS_H_
