#include "ekf_state_index.hpp"
#include "../util/logging.hpp"
#include <set>

namespace odometry {
namespace {
std::size_t findTrackBeginMemoryIndex(const std::vector<KeyFrame> &keyframes, int trackId) {
    std::size_t bestIndex = keyframes.size();
    for (std::size_t i = 0; i < keyframes.size(); ++i) {
        const auto &kf = keyframes[i];
        if (kf.hasFeature(trackId)) {
            bestIndex = i;
        }
    }
    return bestIndex;
}

inline size_t sizeForRandomTrack(double randomTrackSamplingRatio, size_t n) {
    return static_cast<size_t>(std::round(randomTrackSamplingRatio * n));
}
}

int EKFStateIndex::pushHeadKeyframe(int frameNumber, double timestamp) {
    int removedIdx = maxSize() - 1;
    if (keyframes.size() > maxSize() - 1) {
        removedIdx = removeKeyframe();
    }
    KeyFrame kf { frameNumber, timestamp, {} };
    keyframes.insert(keyframes.begin(), kf);
    return removedIdx;
}

void EKFStateIndex::popHeadKeyframe() {
    assert(!keyframes.empty());
    keyframes.erase(keyframes.begin());
    assert(!keyframes.empty());
    // the updated features will be written shortly
    keyframes.begin()->features.clear();
}

float EKFStateIndex::trackScore(int trackId, TrackSampling selection) const {
    size_t length = 0;
    float score = 0;
    const size_t startIndex = selection == TrackSampling::GAP
        ? findTrackBeginMemoryIndex(keyframes, trackId)
        : std::numeric_limits<size_t>::max();

    const Feature *prevFeature = nullptr;
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto &kf = keyframes[i];
        const auto item = kf.features.find(trackId);
        // Assume tracks have no gaps.
        if (item == kf.features.end()) break;
        const auto &feature = item->second;

        bool useThis = false;
        if (selection == TrackSampling::ALL) {
            useThis = true;
        }
        else if (selection == TrackSampling::GAP) {
            if (!feature.usedForVisualUpdate || i == startIndex) {
                useThis = true;
            }
        }
        else if (selection == TrackSampling::RANDOM) {
            if (!feature.usedForVisualUpdate) useThis = true;
        }
        else assert(false && "no implementation for TrackSampling");

        if (useThis) {
            length++;
            // The L1 norm is faster and should be OK for this heuristic
            // also only count the left image point
            if (prevFeature)
                score += (feature.frames[0].imagePoint - prevFeature->frames[0].imagePoint).lpNorm<1>();
        }

        // outside the above if on purpose
        prevFeature = &feature;
    }

    if (selection == TrackSampling::RANDOM) {
        // note: score not implemented properly in this case
        return float(sizeForRandomTrack(parameters.randomTrackSamplingRatio, length));
    }
    return score;
}

// Not `const` because using `tmpIndex`.
void EKFStateIndex::createTrackIndex(
    int trackId,
    std::vector<int> &index,
    TrackSampling selection,
    std::mt19937 &rng
) {
    index.clear();
    tmpIndex.clear();
    const size_t startIndex = selection == TrackSampling::GAP
        ? findTrackBeginMemoryIndex(keyframes, trackId)
        : std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto &kf = keyframes[i];
        const auto item = kf.features.find(trackId);
        // Assume tracks have no gaps.
        if (item == kf.features.end()) break;

        if (selection == TrackSampling::ALL) {
            index.push_back(i);
        }
        else if (selection == TrackSampling::GAP) {
            if (!item->second.usedForVisualUpdate || i == startIndex) {
                index.push_back(i);
            }
        }
        else if (selection == TrackSampling::RANDOM) {
            if (!item->second.usedForVisualUpdate) tmpIndex.push_back(i);
        }
        else assert(false && "no implementation for TrackSampling");
    }

    if (selection == TrackSampling::RANDOM) {
        const size_t n = sizeForRandomTrack(parameters.randomTrackSamplingRatio, tmpIndex.size());
        // Could also use `util::shuffleDeterministic()`.
        for (size_t i = 0; i < n; ++i) {
            size_t ind = rng() % tmpIndex.size();
            index.push_back(tmpIndex.at(ind));
            if (tmpIndex.size() > 1) {
                tmpIndex[ind] = tmpIndex[tmpIndex.size() - 1];
            }
            tmpIndex.resize(tmpIndex.size() - 1);
        }
        // Always use the first point to quickly optimize current pose estimate.
        bool hasFirst = false;
        for (size_t i : index) {
            if (i == 0) {
                hasFirst = true;
                break;
            }
        }
        if (!hasFirst) index[0] = 0;
        // Visualization are terrible without sorting.
        std::sort(index.begin(), index.end());
    }

    assert(index.size() >= 0 && index.size() <= keyframes.size());
}

void EKFStateIndex::createFullIndex(std::vector<int> &index) const {
    index.clear();
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        index.push_back(i);
    }
}

void EKFStateIndex::markTrackUsed(
    int trackId,
    const std::vector<int> &index,
    TrackSampling selection
) {
    if (selection == TrackSampling::GAP) {
        // Mark all.
        for (auto &kf : keyframes) {
            auto item = kf.features.find(trackId);
            if (item != kf.features.end()) {
                item->second.usedForVisualUpdate = true;
            }
        }
    }
    else if (selection == TrackSampling::RANDOM) {
        for (size_t i : index) {
            KeyFrame &kf = keyframes.at(i);
            auto item = kf.features.find(trackId);
            assert(item != kf.features.end());
            item->second.usedForVisualUpdate = true;
        }
    }
    else {
        assert(selection == TrackSampling::ALL); // Do nothing.
    }
}

bool EKFStateIndex::getCurrentTrackPixelCoordinates(int trackId, Eigen::Vector2f &imagePoint) const {
    if (keyframes.size() <= 1) return false;
    const auto &feat = keyframes.at(1).features;
    const auto it = feat.find(trackId);
    if (it == feat.end()) return false;
    imagePoint = it->second.frames[0].imagePoint.cast<float>();
    return true;
}

void EKFStateIndex::buildTrackVectors(
    int trackId,
    const std::vector<int> &index,
    vecVector2d &imageFeatures,
    vecVector2d &featureVelocities,
    Eigen::VectorXd &y,
    bool stereo
) const {
    imageFeatures.clear();
    featureVelocities.clear();
    const std::size_t nValid = index.size();

    y = Eigen::VectorXd::Zero(nValid * 2 * (stereo ? 2 : 1));
    const size_t frameCount = stereo ? 2 : 1;
    size_t ind = 0;
    for (size_t frameInd = 0; frameInd < frameCount; ++frameInd) {
        for (std::size_t j = 0; j < index.size(); ++j) {
            const auto &frame = keyframes.at(index[j]).features.find(trackId)->second.frames[frameInd];
            // Construct features in stacked form.
            y(ind++) = frame.normalizedImagePoint.x();
            y(ind++) = frame.normalizedImagePoint.y();
            imageFeatures.push_back(frame.normalizedImagePoint);
            featureVelocities.push_back(frame.normalizedVelocity);
        }
    }
    assert(featureVelocities.size() == imageFeatures.size());
}

void EKFStateIndex::prune() {
    const auto &kfRef = headKeyFrame();

    // prune map points
    for (int &mapPointTrackId : mapPoints) {
        if (!kfRef.features.count(mapPointTrackId)) mapPointTrackId = -1;
    }

    // prune features & keyframes
    for (std::size_t i = 1; i < keyframes.size(); ++i) {
        auto &features = keyframes.at(i).features;
        auto featureItr = features.begin();
        while (featureItr != features.end()) {
            if (kfRef.features.count(featureItr->first)) featureItr++;
            else featureItr = features.erase(featureItr);
        }
        if (features.empty()) {
            // clear rest
            while (++i < keyframes.size()) keyframes.at(i).features.clear();
            return;
        }
    }
}

int EKFStateIndex::removeKeyframe() {
    assert(!keyframes.empty());
    int removedIdx = -1;
    if (!parameters.cameraTrailFixedScheme)
        // if there are free slots, discard last kf (which is empty & unused)
        for (int i = 1; i < int(keyframes.size()); ++i) {
            if (keyframes.at(i).features.empty()) {
                removedIdx = maxSize() - 1;
                break;
            }
        }

    // Otherwise apply FIFO + strided FIFO + "Tower of Hanoi" backup rotation scheme,
    // where the latest poses j=K,K+1,..,N are updated every STRIDE * 2^{j-K} frames
    // and poses in the middle (ths strided part) are updated every STRIDE frames
    if (removedIdx < 0) {
        frameCounter++;
        const int stride = parameters.cameraTrailStridedLength > 0 ? parameters.cameraTrailStridedStride : 1;
        if (frameCounter % stride != 0) {
            const int firstNonStrided = maxSize() - 1 - parameters.cameraTrailStridedLength - parameters.cameraTrailHanoiLength - 1;
            assert(firstNonStrided > 1);
            removedIdx = firstNonStrided;
        } else {
            const int hanoiCounter = frameCounter / stride;
            removedIdx = maxSize() - 1;
            for (int i = 0; i < parameters.cameraTrailHanoiLength; ++i) {
                if ((hanoiCounter >> i) & 0x1) {
                    removedIdx = maxSize() - 1 - parameters.cameraTrailHanoiLength + i;
                    break;
                }
            }
        }
    }

    assert(removedIdx < int(keyframes.size()));
    keyframes.erase(keyframes.begin() + removedIdx);
    return removedIdx;
}

int EKFStateIndex::offerMapPoint(int trackId) {
    for (int i = 0; i < int(mapPoints.size()); ++i) {
        int &mapPointTrackId = mapPoints.at(i);
        if (mapPointTrackId == -1) {
            mapPointTrackId = trackId;
            return i;
        }
    }
    return -1;
}

void EKFStateIndex::extract3DFeatures(
    int trackId,
    const std::vector<int> &index,
    CameraPoseTrail &camPoseTrail) const
{
    const bool isStereo = index.size() != camPoseTrail.size();
    if (!isStereo) return;

    for (std::size_t i = 0; i < index.size(); ++i) {
        auto &firstCamPose = camPoseTrail.at(i);
        const auto &feature = keyframes.at(index.at(i)).features.find(trackId)->second;

        firstCamPose.hasFeature3D = true;
        firstCamPose.feature3DIdp = feature.triangulatedStereoPointIdp;
        firstCamPose.feature3DCov = feature.triangulatedStereoCov;
    }
}

bool EKFStateIndex::widestBaseline(
    int trackId,
    size_t &keyframe0,
    size_t &keyframe1,
    Eigen::Vector2d &imagePoint0,
    Eigen::Vector2d &imagePoint1) const {
    const auto poseCount = keyframes.size();
    if (poseCount < 2) return false;

    bool ok = false;
    for (keyframe0 = 0; keyframe0 < poseCount; ++keyframe0) {
        if (keyframes[keyframe0].features.count(trackId)) {
            imagePoint0 = keyframes[keyframe0].features.at(trackId).frames[0].normalizedImagePoint;
            ok = true;
            break;
        }
    }
    if (!ok) return false;
    for (keyframe1 = poseCount - 1; ; --keyframe1) {
        if (keyframes[keyframe1].features.count(trackId)) {
            imagePoint1 = keyframes[keyframe1].features.at(trackId).frames[0].normalizedImagePoint;
            break;
        }
        if (keyframe1 == 0) break; // size_t >= 0 is always true.
    }
    assert(keyframe1 >= keyframe0);
    return keyframe0 != keyframe1;
}

void EKFStateIndex::getVisualizationTracks(VisualizationTrackCollection &tracks) const {
    tracks.clear();
    if (keyframes.empty()) return;
    std::set<int> currentIds;
    for (const auto &it : keyframes[0].features) {
        currentIds.insert(it.first);
    }
    for (const KeyFrame &keyframe : keyframes) {
        for (const auto &it : keyframe.features) {
            if (!tracks.count(it.first)) {
                tracks.emplace(it.first, VisualizationTrack {
                    .points = {},
                    .active = currentIds.count(it.first) > 0,
                });
            }
            tracks[it.first].points.push_back(it.second.frames[0].imagePoint);
        }
    }
}

void EKFStateIndex::updateVelocities(int trackId) {
    if (keyframes.size() < 2) return;
    if (keyframes[0].timestamp <= keyframes[1].timestamp) return;
    if (!keyframes[0].features.count(trackId) || !keyframes[1].features.count(trackId)) return;
    Feature &feature0 = keyframes[0].features.at(trackId);
    Feature &feature1 = keyframes[1].features.at(trackId);

    for (size_t i : {0, 1}) {
        Feature::Frame &f0 = feature0.frames[i];
        Feature::Frame &f1 = feature1.frames[i];
        Eigen::Vector2d v = (f0.normalizedImagePoint - f1.normalizedImagePoint)
            / (keyframes[0].timestamp - keyframes[1].timestamp);
        f0.normalizedVelocity = v;
        if (keyframes.size() == 2 || !keyframes[2].features.count(trackId)) {
            f1.normalizedVelocity = v;
        }
        else {
            if (keyframes[0].timestamp <= keyframes[2].timestamp) return;
            Feature::Frame &f2 = keyframes[2].features.at(trackId).frames[i];
            f1.normalizedVelocity = (f0.normalizedImagePoint - f2.normalizedImagePoint)
                / (keyframes[0].timestamp - keyframes[2].timestamp);
        }
    }
}

} // namespace odometry
