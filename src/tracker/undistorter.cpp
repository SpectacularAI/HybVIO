#include <accelerated-arrays/function.hpp>
#include <accelerated-arrays/standard_ops.hpp>
#include <accelerated-arrays/cpu/image.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>
#include <accelerated-arrays/standard_ops.hpp>

#include "camera.hpp"
#include "gpu_util.hpp"
#include "undistorter.hpp"
#include "../odometry/parameters.hpp"
#include "../util/allocator.hpp"
#include "../api/vio.hpp"

namespace tracker {
namespace {
std::string undistortGlslGpu(const Camera &rectifiedCamera, const Camera &origCamera) {
    std::ostringstream oss;
    oss << rectifiedCamera.pixelToRayGlsl() << "\n"
        << origCamera.rayToPixelGlsl() << "\n"
        << R"(
        void main() {
            ivec2 coord = ivec2(v_texCoord * vec2(u_outSize));
            vec2 rectifiedCoord = vec2(coord);
            vec2 origCoord = rayToPixel(pixelToRay(rectifiedCoord));
            vec2 origCoordTex = (origCoord + vec2(0.5f)) / vec2(textureSize(u_texture, 0));
            outValue = texture(u_texture, origCoordTex).r;
        }
        )";
    return oss.str();
}

accelerated::operations::Function buildUndistortOp(
    std::shared_ptr<const Camera> camera,
    std::shared_ptr<const Camera> rectifiedCamera,
    accelerated::Image::Factory &ifac,
    accelerated::operations::StandardFactory &ofac)
{
    auto imgSpec = ifac.getSpec(1, accelerated::ImageTypeSpec::DataType::UFIXED8);
    if (imgSpec.storageType == accelerated::ImageTypeSpec::StorageType::CPU) return {};
    return gpu_util::wrapShader(ofac, undistortGlslGpu(*rectifiedCamera, *camera), { imgSpec }, imgSpec);
}

class UndistorterImplementation : public Undistorter {
private:
    accelerated::Image::Factory &ifac;
    accelerated::operations::StandardFactory &ofac;
    util::Allocator<accelerated::Image> imageAllocator;
    std::shared_ptr<const Camera> originalCamera, undistortedCamera;
    accelerated::operations::Function undistortGpuOp;

    bool undistortCpu(const Eigen::Vector2d &pixRect, Eigen::Vector2d &pixOrig, const Camera &rectifiedCamera, const Camera &origCamera) {
        Eigen::Vector3d ray;
        if (!rectifiedCamera.pixelToRay(pixRect, ray)) return false;
        return origCamera.rayToPixel(ray, pixOrig);
    }

public:
    UndistorterImplementation(
        int width, int height,
        std::shared_ptr<const Camera> camera,
        accelerated::Image::Factory &ifac,
        accelerated::operations::StandardFactory &ofac)
    :
        ifac(ifac), ofac(ofac),
        imageAllocator([&ifac, width, height]() {
            return ifac.create(width, height, 1, accelerated::ImageTypeSpec::DataType::UFIXED8);
        }),
        undistortedCamera(camera)
    {}

    Result undistort(accelerated::Image &image, std::shared_ptr<const Camera> camera) {
        auto outImage = imageAllocator.next();
        constexpr bool INTERPOLATE = true;

        if (image.storageType == accelerated::Image::StorageType::CPU) {
            auto input = accelerated::opencv::ref(image);
            auto output = accelerated::opencv::ref(*outImage);

            const int w = input.cols;
            const int h = input.rows;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    Eigen::Vector2d pixRect(x, y), pixOrig;
                    float out = 0;
                    if (undistortCpu(pixRect, pixOrig, *undistortedCamera, *camera)) {
                        if (pixOrig(0) >= 0 && pixOrig(0) < w && pixOrig(1) >= 0 && pixOrig(1) < h) {
                            if (INTERPOLATE) {
                                const int x0 = int(std::floor(pixOrig(0))), y0 = int(std::floor(pixOrig(1)));
                                const float xfrac = pixOrig(0) - x0, yfrac = pixOrig(1) - y0;
                                for (int iy = 0; iy < 2; ++iy) {
                                    const float wy = iy > 0 ? yfrac : (1 - yfrac);
                                    for (int ix = 0; ix < 2; ++ix) {
                                        const float wx = ix > 0 ? xfrac : (1 - xfrac);
                                        out += input.at<std::uint8_t>(y0 + iy, x0 + ix) * wx * wy;
                                    }
                                }
                            } else {
                                out = input.at<std::uint8_t>(int(pixOrig(1)), int(pixOrig(0)));
                            }
                        }
                    }
                    output.at<std::uint8_t>(y, x) = int(out + 0.5);
                }
            }

            return {
                .camera = undistortedCamera,
                .image = outImage,
                .future = accelerated::Future::instantlyResolved()
            };
        } else {
            if (originalCamera) {
                const double focalDiff = originalCamera->getFocalLength() - camera->getFocalLength();
                if (focalDiff > 1e-6) {
                    log_warn("Per-frame camera parameters ignored in GPU undistortion");
                }
            } else {
                originalCamera = camera;
                undistortGpuOp = buildUndistortOp(originalCamera, undistortedCamera, ifac, ofac);
            }
            constexpr auto BORDER_TYPE = accelerated::Image::Border::CLAMP;
            constexpr auto INTERPOLATION = INTERPOLATE ?
                accelerated::Image::Interpolation::LINEAR :
                accelerated::Image::Interpolation::NEAREST;

            gpu_util::setGlBorder(image, BORDER_TYPE);
            gpu_util::setGlInterpolation(image, INTERPOLATION);
            return {
                .camera = undistortedCamera,
                .image = outImage,
                .future = accelerated::operations::callUnary(undistortGpuOp, image, *outImage)
            };
        }
    }
};
}

std::unique_ptr<Undistorter> Undistorter::buildRectified(
    int w, int h,
    std::shared_ptr<const Camera> rectifiedCamera,
    accelerated::Image::Factory &ifac,
    accelerated::operations::StandardFactory &ofac,
    const odometry::ParametersTracker &p)
{
    if (!p.useRectification) return {};
    return std::unique_ptr<Undistorter>(new UndistorterImplementation(w, h, rectifiedCamera, ifac, ofac));
}

std::unique_ptr<Undistorter> Undistorter::buildMono(
    int w, int h,
    float focalLength,
    accelerated::Image::Factory &ifac,
    accelerated::operations::StandardFactory &ofac,
    const odometry::ParametersTracker &p)
{
    if (!p.useRectification) return {};

    api::CameraParameters i;
    // equal focal lengths and pp in the center of image (feature, not bug)
    i.focalLengthX = focalLength * p.rectificationZoom;
    i.focalLengthY = focalLength * p.rectificationZoom;
    i.principalPointX = w * 0.5f;
    i.principalPointY = h * 0.5f;
    std::shared_ptr<const Camera> camera = Camera::buildPinhole(i, {}, w, h, nullptr);

    return std::unique_ptr<Undistorter>(new UndistorterImplementation(w, h, camera, ifac, ofac));
}

Undistorter::~Undistorter() = default;
}
