#ifndef TRACKER_GPU_UTIL_H_
#define TRACKER_GPU_UTIL_H_

#include <accelerated-arrays/standard_ops.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

namespace tracker {
namespace gpu_util {
/** Wraps shader to Function if compiled with GPU support. Otherwise aborts. */
accelerated::operations::Function wrapShader(
    accelerated::operations::StandardFactory &ops,
    const std::string &shaderBody,
    const std::vector<accelerated::ImageTypeSpec> &inputs,
    const accelerated::ImageTypeSpec &output);

void setGlInterpolation(accelerated::Image &image, accelerated::Image::Interpolation interpolation);
void setGlBorder(accelerated::Image &image, accelerated::Image::Border border);
}
}

#endif
