#ifndef VIO_TYPE_CONVERT_HPP
#define VIO_TYPE_CONVERT_HPP

#include <string>
#include "vio.hpp"
#include "types.hpp"

#include <Eigen/Dense>

namespace api {

Eigen::Vector3d vectorToEigen(api::Vector3d v);
api::Vector3d eigenToVector(const Eigen::Vector3d &v);
Eigen::Vector4d quaternionToEigenVector(api::Quaternion v);
Eigen::Quaterniond quaternionToEigen(api::Quaternion v);
api::Quaternion eigenVectorToQuaternion(const Eigen::Vector4d &orientation);
api::Quaternion eigenToQuaternion(const Eigen::Quaterniond &orientation);
api::Pose eigenToPose(double t, const Eigen::Vector3d &position, const Eigen::Vector4d &orientation);
api::Matrix3d eigenToMatrix(const Eigen::Matrix3d &T);
std::string outputToJson(const api::VioApi::VioOutput &out, bool withTail);

} // namespace api

#endif // VIO_UTIL_HPP
