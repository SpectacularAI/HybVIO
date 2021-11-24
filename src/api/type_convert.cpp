#include "type_convert.hpp"
#include <nlohmann/json.hpp>
#include "internal.hpp"

namespace api {

Eigen::Vector3d vectorToEigen(api::Vector3d v) {
    return Eigen::Vector3d(v.x, v.y, v.z);
}

api::Vector3d eigenToVector(const Eigen::Vector3d &v) {
    return { .x = v.x(), .y = v.y(), .z = v.z() };
}

Eigen::Vector4d quaternionToEigenVector(api::Quaternion v) {
    return Eigen::Vector4d(v.w, v.x, v.y, v.z);
}

Eigen::Quaterniond quaternionToEigen(api::Quaternion v) {
    return Eigen::Quaterniond(v.w, v.x, v.y, v.z);
}

api::Quaternion eigenVectorToQuaternion(const Eigen::Vector4d &orientation) {
    return {
        .x = orientation(1),
        .y = orientation(2),
        .z = orientation(3),
        .w = orientation(0)
    };
}

api::Quaternion eigenToQuaternion(const Eigen::Quaterniond &orientation) {
    return {
        .x = orientation.x(),
        .y = orientation.y(),
        .z = orientation.z(),
        .w = orientation.w()
    };
}

api::Pose eigenToPose(double t, const Eigen::Vector3d &position, const Eigen::Vector4d &orientation) {
    return {
        .time = t,
        .position = api::eigenToVector(position),
        .orientation = api::eigenVectorToQuaternion(orientation),
    };
}

api::Matrix3d eigenToMatrix(const Eigen::Matrix3d &T) {
    Matrix3d ret;
    for (std::size_t i = 0; i < ret.size(); ++i) {
        auto &row = ret[i];
        for (std::size_t j = 0; j < row.size(); ++j) {
            row[j] = T(i, j);
        }
    }
    return ret;
}

nlohmann::json vectorToJson(const api::Vector3d &vec) {
    nlohmann::json j = {{"x", vec.x}, {"y", vec.y}, {"z", vec.z}};
    return j;
}

nlohmann::json quaternionToJson(const api::Quaternion &quat) {
    nlohmann::json j = {{"w", quat.w}, {"x", quat.x}, {"y", quat.y}, {"z", quat.z}};
    return j;
}

std::string outputToJson(const api::VioApi::VioOutput &out, bool withTail) {
    const api::Pose &pose = out.pose;
    const auto &poseTrail = out.poseTrail;
    const auto &internal = reinterpret_cast<const api::InternalAPI::Output&>(out);

    nlohmann::json outputJson;

    if (!internal.additionalData.empty()) {
        outputJson.update(nlohmann::json::parse(internal.additionalData));
    }

    outputJson["time"] = out.pose.time;
    outputJson["position"] = api::vectorToJson(pose.position);
    outputJson["orientation"] = api::quaternionToJson(pose.orientation);
    outputJson["velocity"] = api::vectorToJson(out.velocity);

    if (withTail) {
        outputJson["poseTrail"] = nlohmann::json::array();
        for (const auto &p : poseTrail) {
            outputJson["poseTrail"].push_back({
                {"position", api::vectorToJson(p.position)},
                {"orientation", api::quaternionToJson(p.orientation)}
            });
        }

    }

    return outputJson.dump();
}

} // namespace api
