#include "gpu_util.hpp"
#include "../util/logging.hpp"

#ifdef DAZZLING_GPU_ENABLED
#include <accelerated-arrays/opengl/image.hpp>
#include <accelerated-arrays/opengl/operations.hpp>
#endif

namespace tracker {
namespace gpu_util {

accelerated::operations::Function wrapShader(
    accelerated::operations::StandardFactory &ops,
    const std::string &shaderBody,
    const std::vector<accelerated::ImageTypeSpec> &inputs,
    const accelerated::ImageTypeSpec &output)
{
    assert(output.storageType != accelerated::Image::StorageType::CPU);
#ifdef DAZZLING_GPU_ENABLED
    auto &gpuOps = reinterpret_cast<accelerated::opengl::operations::Factory&>(ops);
    return gpuOps.wrapShader(shaderBody, inputs, output);
#else
    (void)ops; (void)shaderBody; (void)inputs;
    assert(false && "no GPU support");
    return {};
#endif
}

void setGlInterpolation(accelerated::Image &image, accelerated::Image::Interpolation interpolation) {
#ifdef DAZZLING_GPU_ENABLED
    accelerated::opengl::Image::castFrom(image).setInterpolation(interpolation);
#else
    (void)image; (void)interpolation;
    assert(false && "no GPU support");
#endif
}

void setGlBorder(accelerated::Image &image, accelerated::Image::Border border) {
#ifdef DAZZLING_GPU_ENABLED
    accelerated::opengl::Image::castFrom(image).setBorder(border);
#else
    (void)image; (void)border;
    assert(false && "no GPU support");
#endif
}

}
}
