#include <algorithm>
#include <cmath>
#include <sstream>
#include <cassert>
#include <accelerated-arrays/standard_ops.hpp>
#include <accelerated-arrays/cpu/operations.hpp>
#include <accelerated-arrays/opencv_adapter.hpp>

#include "feature_detector.hpp"
#include "../util/logging.hpp"
#include "../odometry/parameters.hpp"
#include "image.hpp"
#include "gpu_util.hpp"

// required for fast CPU fallbacks
#include <opencv2/opencv.hpp>

namespace tracker {
using gpu_util::wrapShader;

namespace {

const auto BORDER_TYPE = accelerated::Image::Border::MIRROR;
typedef accelerated::cpu::Image CpuImage;
typedef accelerated::FixedPoint<std::uint8_t> Ufixed8;

struct KeyPoint : public Feature::Point {
    float response;
};

struct Sobel {
    typedef std::int16_t Type;
    static constexpr float scale = 100, bias = 0;

    accelerated::operations::Function opX, opY;
    std::shared_ptr<accelerated::Image> bufX, bufY;

    Sobel(int w, int h,
          accelerated::Image::Factory &images,
          accelerated::operations::StandardFactory &ops)
    {
        accelerated::ImageTypeSpec inputSpec = images.getSpec<Ufixed8, 1>();
        bufX = images.create<Type, 1>(w, h);
        bufY = images.createLike(*bufX);

        // note: signs don't matter here that much
        opX = ops.fixedConvolution2D({
                 {-1, 0, 1},
                 {-2, 0, 2},
                 {-1, 0, 1}
             })
            .scaleKernelValues(scale)
            .setBias(bias)
            .setBorder(BORDER_TYPE)
            .build(inputSpec, *bufX);

        opY = ops.fixedConvolution2D({
                 {-1,-2,-1 },
                 { 0, 0, 0 },
                 { 1, 2, 1 }
             })
            .scaleKernelValues(scale)
            .setBias(bias)
            .setBorder(BORDER_TYPE)
            .build(inputSpec, *bufY);
    }

    void operator()(accelerated::Image &inputBuffer) {
        assert(inputBuffer.dataType == accelerated::Image::DataType::UFIXED8);
        accelerated::operations::callUnary(opX, inputBuffer, *bufX);
        accelerated::operations::callUnary(opY, inputBuffer, *bufY);
    }

    std::function<void()> debugRenderer(accelerated::Image &outBuffer, accelerated::operations::StandardFactory &ops) {
        const float renderScale = 1 / 255.0;
        const float renderBias = -bias / renderScale + 0.5f;
        auto renderOp = ops.affineCombination()
                .addLinearPart({ {1}, {0}, {0}, {0} })
                .addLinearPart({ {0}, {1}, {0}, {0} })
                .scaleLinearValues(renderScale)
                .setBias({ renderBias, renderBias, 0.5, 1 })
                .build(*bufX, outBuffer);

        return [this, renderOp, &outBuffer]() {
            accelerated::operations::callBinary(renderOp, *bufX, *bufY, outBuffer);
        };
    }
};

struct StructureMatrix {
    typedef std::int16_t Type;
    static constexpr float scale = 5000, bias = 0;
    static constexpr const char * glslType = "ivec4"; // could expose these in accelerated-arrays

    Sobel &sobel;
    accelerated::operations::Function op;
    std::shared_ptr<accelerated::Image> buffer;

    static std::string shaderBody() {
        std::ostringstream oss;
        oss << "const float inScaleInv = float(" << (1.0 / Sobel::scale) << ");\n"
            << "const float inBias = float(" << Sobel::bias << ");\n"
            << "const float outScale = float(" << scale << ");\n"
            << "const float outBias = float(" << bias << ");\n"
            << R"(
            void main() {
                ivec2 coord = ivec2(v_texCoord * vec2(u_outSize));
                vec2 der = (vec2(
                    float(texelFetch(u_texture1, coord, 0).r),
                    float(texelFetch(u_texture2, coord, 0).r)) - inBias) * inScaleInv;
                vec2 d2 = der*der;
                float dxdy = der.x*der.y;
                outValue = )" << glslType << R"((vec4(
                    d2.x, dxdy,
                    dxdy, d2.y
                ) * outScale + outBias);
            }
            )";
        return oss.str();
    }

    StructureMatrix(Sobel &sobelInput,
        accelerated::Image::Factory &images,
        accelerated::operations::StandardFactory &ops)
    :
        sobel(sobelInput)
    {
        buffer = images.create<Type, 4>(sobel.bufX->width, sobel.bufX->height);
        op = wrapShader(ops, shaderBody(), { *sobel.bufX, *sobel.bufY }, *buffer);
    }

    void operator()(accelerated::Image &inputBuffer) {
        sobel(inputBuffer);
        accelerated::operations::callBinary(op, *sobel.bufX, *sobel.bufY, *buffer);
    }
};

struct FilteredStructureMatrix {
    StructureMatrix &input;
    accelerated::operations::Function filterX, filterY;
    std::shared_ptr<accelerated::Image> buffer;

    FilteredStructureMatrix(StructureMatrix &input,
        accelerated::Image::Factory &images,
        accelerated::operations::StandardFactory &ops,
        const odometry::ParametersTracker &parameters)
    :
            input(input)
    {
        buffer = images.createLike(*input.buffer);

        std::vector<std::vector<double>> kernelX, kernelY;
        kernelX.push_back({});
        for (int i = 0; i < parameters.gfttBlockSize; ++i) {
            kernelX.at(0).push_back(1);
            kernelY.push_back({ 1 });
        }
        const float kernelScale = 1 / double(parameters.gfttBlockSize);

        // separable box filter. For large values of gfttBlockSize
        // not the most efficient implementation for large gfttBlockSizes
        filterX = ops.fixedConvolution2D(kernelX)
                .scaleKernelValues(kernelScale)
                .setBorder(BORDER_TYPE)
                .build(*buffer);

        filterY = ops.fixedConvolution2D(kernelY)
                .scaleKernelValues(kernelScale)
                .setBorder(BORDER_TYPE)
                .build(*buffer);
    }

    accelerated::Image &getOutputBuffer() const {
        return *input.buffer;
    }

    void operator()(accelerated::Image &inputBuffer) {
        input(inputBuffer);
        accelerated::operations::callUnary(filterX, *input.buffer, *buffer);
        accelerated::operations::callUnary(filterY, *buffer, *input.buffer);
    }
};

struct CornerResponse {
    static constexpr bool USE_HARRIS = false;

    virtual ~CornerResponse() = default;
    virtual accelerated::Image& operator()(accelerated::Image &inputBuffer) = 0;
    virtual std::function<void()> debugRenderer(accelerated::Image &outBuffer, accelerated::operations::StandardFactory &ops) = 0;
};

struct GpuCornerResponse : CornerResponse {
    Sobel sobel;
    StructureMatrix structureMatrix;
    FilteredStructureMatrix filteredStructureMatrix;

    static constexpr bool VISUALIZE_SOBEL = false;

    GpuCornerResponse(int w, int h,
        accelerated::Image::Factory &images,
        accelerated::operations::StandardFactory &ops,
        const odometry::ParametersTracker &parameters)
    :
        sobel(w, h, images, ops),
        structureMatrix(sobel, images, ops),
        filteredStructureMatrix(structureMatrix, images, ops, parameters)
    {
        auto &inBuf = filteredStructureMatrix.getOutputBuffer();
        buffer = images.create<Type, 1>(inBuf.width, inBuf.height);
        op = wrapShader(ops, shaderBody(parameters), { inBuf }, *buffer);
    }

    typedef float Type;
    static constexpr float scale = 1.0, bias = 0.0;
    //static constexpr float scale = 3.0, bias = 0.5;

    accelerated::operations::Function op;
    std::shared_ptr<accelerated::Image> buffer;

    static std::string shaderBody(const odometry::ParametersTracker &parameters) {
        std::ostringstream oss;
        oss << "const float inScaleInv = float(" << (1.0 / StructureMatrix::scale) << ");\n"
            << "const float inBias = float(" << StructureMatrix::bias << ");\n"
            << "const float outBias = float(" << bias << ");\n"
            << "const float outScale = float(" << scale << ");\n"
            << "const float minResponse = float(" << parameters.gfttMinResponse << ");\n"
            << R"(
            void main() {
                ivec2 coord = ivec2(v_texCoord * vec2(u_outSize));
                vec4 m = (vec4(texelFetch(u_texture, coord, 0)) - inBias) * inScaleInv;
                float det = m.x * m.a - m.y * m.z;
                float tr = m.x + m.y;
                )";
        if (USE_HARRIS) {
            oss << R"(
                const float k = float()" << parameters.gfttK << R"();
                float result = det - k * tr;
                )";
        } else {
            // cf. http://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/G2dmatrices/index.html
            oss << R"(
                float halfTr = 0.5*tr;
                float result = halfTr - sqrt(max(halfTr*halfTr - det, 0.0));
                )";
        }
        oss << R"(
                if (result < minResponse) {
                    outValue = -1.0;
                } else {
                    outValue = result*outScale + outBias;
                }
            }
        )";
        return oss.str();
    }

    accelerated::Image& operator()(accelerated::Image &inputBuffer) final {
        filteredStructureMatrix(inputBuffer);
        accelerated::operations::callUnary(op, filteredStructureMatrix.getOutputBuffer(), *buffer);
        return *buffer;
    }

    std::function<void()> debugRenderer(accelerated::Image &outBuffer, accelerated::operations::StandardFactory &ops) final {
        if (VISUALIZE_SOBEL) {
            return sobel.debugRenderer(outBuffer, ops);
        } else {
            auto renderOp = ops.pixelwiseAffine({ {1}, {1}, {1}, {0} })
                    .setBias({ 0, 0, 0, 1 })
                    .build(*buffer, outBuffer);

            return [this, renderOp, &outBuffer]() {
                accelerated::operations::callUnary(renderOp, *buffer, outBuffer);
            };
        }

    }
};

struct CpuCornerResponse : CornerResponse {
    const odometry::ParametersTracker &parameters;
    cv::Mat response, tmp;
    std::unique_ptr<CpuImage> responseAcc;
    static constexpr double GAIN = 16.0;

    CpuCornerResponse(const odometry::ParametersTracker &p) : parameters(p) {}

    accelerated::Image& operator()(accelerated::Image &inputBuffer) final {
        cv::Mat mat = accelerated::opencv::ref(inputBuffer);
        constexpr int SOBEL_K_SIZE = 3; // "Aperture parameter for the Sobel operator."

        if (USE_HARRIS) {
            cv::cornerHarris(mat, response, parameters.gfttBlockSize, SOBEL_K_SIZE, parameters.gfttK);
        } else {
            cv::cornerMinEigenVal(mat, response,  parameters.gfttBlockSize, SOBEL_K_SIZE); // = GFTT
        }

        if (!responseAcc) {
            responseAcc = accelerated::opencv::ref(response);
        }

        return *responseAcc;
    }

    std::function<void()> debugRenderer(accelerated::Image &outBuffer, accelerated::operations::StandardFactory &ops) final {
        (void)ops;
        return [this, &outBuffer]() {
            if (response.empty()) return;

            constexpr double SCALE = 256 * GAIN;
            constexpr int MID = 0;
            response.convertTo(tmp, CV_8UC1, SCALE, MID);
            cv::cvtColor(tmp, accelerated::opencv::ref(outBuffer), cv::COLOR_GRAY2BGRA);
        };
    }
};

struct CollectMax {
    /*typedef float Type;
    static constexpr const char * glslType = "vec4";
    static constexpr float scale = 1.0, bias = 0;
    static constexpr float responseScale = 1 / 10.0;*/

    typedef std::uint16_t Type;
    static constexpr const char * glslType = "uvec4";
    static constexpr float scale = 1000.0, bias = 100;
    static constexpr float responseScale = 1 / scale;

    const odometry::ParametersTracker &parameters;
    std::vector<int> reduceFactors;

    std::vector<accelerated::operations::Function> opList;
    std::vector<std::shared_ptr<accelerated::Image>> buffers;

    // TODO: investigate in which cases this is really necessary and/or
    // implement automatic read_adapter in accelerated-arrays
    std::shared_ptr<accelerated::Image> outRgba8;
    accelerated::operations::Function outRgba8Op;

    std::vector<std::uint8_t> featureData;

    int blockSize() const {
        int b = 1;
        for (auto r : reduceFactors) b *= r;
        return b;
    }

    static std::string shaderBody(bool xdir,
            int reduceFactor, int curBlockSize,
            float curScale, float curBias) {
        std::ostringstream oss;

        oss << "#define CUR_BLOCK_SIZE " << curBlockSize << "\n"
            << "#define REDUCE_FACTOR " << reduceFactor << "\n";

        if (xdir) {
            oss << "const ivec2 delta = ivec2(CUR_BLOCK_SIZE, 0);\n";
            oss << "const ivec2 reduceSize = ivec2(REDUCE_FACTOR, 1);\n";
            oss << "const ivec2 inDelta = ivec2(1, 0);\n";
        } else {
            oss << "const ivec2 delta = ivec2(0, CUR_BLOCK_SIZE);\n";
            oss << "const ivec2 reduceSize = ivec2(1, REDUCE_FACTOR);\n";
            oss << "const ivec2 inDelta = ivec2(0, 1);\n";
        }

        oss << "const float scale = float(" << curScale << ");\n"
            << "const float bias = float(" << curBias << ");\n"
            << "const float MIN_VAL = -1e10;\n" // NOTE: could have corner cases
            << R"(
            void main() {
                ivec2 coord = ivec2(v_texCoord * vec2(u_outSize)) * reduceSize;
                ivec2 inSize = textureSize(u_texture, 0);
                vec4 best = vec4(MIN_VAL, 0, 0, 0);
                for (int i = 0; i < REDUCE_FACTOR; ++i) {
                    if (coord.x >= inSize.x || coord.y >= inSize.y) break;
                    vec4 v = vec4(texelFetch(u_texture, coord, 0));
                    float val = v.x * scale + bias;
                    if (val > best.x) {
                        best = vec4(
                            val,
                            v.yz + vec2(delta * i),
                            0
                        );
                    }
                    coord += inDelta;
                }
                outValue = )"
            << glslType << "(best);\n"
            << "}\n";
        // log_debug("%s", oss.str().c_str());
        return oss.str();
    }

    void cpuImplementation(const CpuImage &response, int bs, std::vector<KeyPoint> &keyPoints) const {
        keyPoints.clear();
        for (int yBlock = 0; yBlock < std::ceil(response.height / bs); ++yBlock) {
            for (int xBlock = 0; xBlock < std::ceil(response.width / bs); ++xBlock) {
                float maxResponse = -1e10;
                int bestX = 0, bestY = 0;
                for (int y = yBlock * bs; y < (yBlock + 1) * bs && y < response.height; ++y) {
                    for (int x = xBlock * bs; x < (xBlock + 1) * bs && x < response.width; ++x) {
                        const float r = response.get<float>(x, y) * CpuCornerResponse::GAIN;
                        if (r > maxResponse && r > parameters.gfttMinResponse) {
                            bestX = x;
                            bestY = y;
                            maxResponse = r;
                        }
                    }
                }
                KeyPoint kp;
                kp.x = bestX;
                kp.y = bestY;
                // NOTE: not the same scale as in GPU version
                kp.response = maxResponse;
                keyPoints.push_back(kp);
            }
        }
    }

    CollectMax(int width, int height,
        accelerated::Image::Factory &images,
        accelerated::operations::StandardFactory &ops,
        const odometry::ParametersTracker &p,
        bool useGpu)
    :
        parameters(p)
    {
        const int targetBlockSize = int(parameters.gfttMinDistance);

        // TODO: crude
        if (targetBlockSize >= 32) {
            reduceFactors = { 4, 4, 2 };
        } else if (targetBlockSize >= 16) {
            reduceFactors = { 4, 4 };
        } else {
            reduceFactors = { 4, 2 };
        }

        if (!useGpu) return;

        float curScale = scale / GpuCornerResponse::scale;
        float curBias = -GpuCornerResponse::bias / curScale + bias;

        std::shared_ptr<accelerated::Image> inBuf;
        accelerated::ImageTypeSpec inputSpec = images.getSpec<GpuCornerResponse::Type, 1>();

        for (int dir = 0; dir < 2; ++dir) {
            bool xdir = dir == 0;
            int curBlockSize = 1;
            for (int r : reduceFactors) {
                if (xdir) width = (width + r - 1) / r;
                else height = (height + r - 1) / r;
                buffers.push_back(images.create<Type, 4>(width, height));

                opList.push_back(wrapShader(ops,
                        shaderBody(xdir, r, curBlockSize, curScale, curBias),
                        { inBuf ? *inBuf : inputSpec }, *buffers.back()));
                curScale = 1.0;
                curBias = 0.0;
                curBlockSize *= r;
                inBuf = buffers.back();
            }
        }

        auto &features = *buffers.back();
        outRgba8 = images.create<Ufixed8, 4>(features.width, features.height);

        outRgba8Op = ops.pixelwiseAffine({
            {responseScale,0,0,0},
            {0,1/255.,0,0},
            {0,0,1/255.0,0},
            {0,0,0,0}
        }).build(features, *outRgba8);

        featureData.resize(outRgba8->numberOfScalars());

        log_debug("reading %d x %d = %zu features (= %zu bytes) in GpuFeatureDetector, block size %d (target was %d)",
                features.width,
                features.height,
                features.numberOfPixels(),
                features.size(),
                blockSize(),
                targetBlockSize);
    }

    accelerated::Future operator()(accelerated::Processor &proc, accelerated::Image &inputBuffer, std::vector<KeyPoint>& keyPoints) {
        if (buffers.empty()) {
            return proc.enqueue([this, &inputBuffer, &keyPoints]() {
                cpuImplementation(CpuImage::castFrom(inputBuffer), blockSize(), keyPoints);
            });
        }

        accelerated::Image *inBuf = &inputBuffer;
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            accelerated::Image *outBuf = buffers.at(i).get();
            accelerated::operations::callUnary(opList.at(i), *inBuf, *outBuf);
            inBuf = outBuf;
        }

        accelerated::operations::callUnary(outRgba8Op, *buffers.back(), *outRgba8);
        outRgba8->readRawFixedPoint(featureData);
        return proc.enqueue([&keyPoints, this]() {
            keyPoints.clear();
            const int bs = blockSize();
            int idx = 0;
            for (int by = 0; by < outRgba8->height; ++by) {
                for (int bx = 0; bx < outRgba8->width; ++bx) {
                    int response = featureData.at(idx++);
                    int dx = featureData.at(idx++);
                    int dy = featureData.at(idx++);
                    idx++; // ignore channel 4
                    // log_debug("bx %d, by %d, dx %d, dy %d, bs %d response %d", bx, by, dx, dy, bs, response);

                    if (response > 0) {
                        KeyPoint kp;
                        kp.x = dx + bx * bs;
                        kp.y = dy + by * bs;
                        kp.response = response;
                        keyPoints.push_back(kp);
                    }
                }
            }
        });
    }

    std::function<void()> debugRenderer(accelerated::Image &buf, accelerated::operations::StandardFactory &ops) {
        std::ostringstream oss;
        const int bs = blockSize();
        oss << "const ivec2 blockSize = ivec2(" << bs << ", " << bs << ");\n"
            << "const float inScaleInv = float(" << (1.0 / scale) << ");\n"
            << "const float inBias = float(" << bias << ");\n"
            << "const float renderScale = float(10.0);\n"
            << "const float renderBias = float(0);\n"
            << R"(
            void main() {
                ivec2 coord = ivec2(v_texCoord * vec2(u_outSize));
                ivec2 blockCoord = coord / blockSize;
                ivec2 localCoord = coord - blockCoord * blockSize;
                vec4 blockMax = vec4(texelFetch(u_texture, blockCoord, 0));
                float distToMax = length(vec2(localCoord) - blockMax.yz);
                float maxVal = (blockMax.x - inBias) * inScaleInv;

                // visualizations stuff
                const float R = 3.0;
                float maxScaled = maxVal * renderScale + renderBias;
                float nearMax;
                if (distToMax < R) {
                    nearMax = max(maxScaled, 0.1);
                } else {
                    nearMax = 0.0;
                }
                //outValue = vec4(vec3(nearMax), 1);
                //outValue = vec4(blockMax.xy / vec2(blockSize), maxVal, 1);
                outValue = vec4(vec2(localCoord.xy) / vec2(blockSize) * 0.2, nearMax, 1);
            }
            )";

        auto &bufToVisualize = *buffers.back();

        auto renderOp = wrapShader(ops, oss.str(), { bufToVisualize }, buf);
        return [renderOp, &buf, &bufToVisualize]() {
            accelerated::operations::callUnary(renderOp, bufToVisualize, buf);
        };
    }
};

class FeatureDetectorImplementation : public FeatureDetector {
private:
    const int width, height;

    std::unique_ptr<accelerated::Processor> cpuProcessor;
    accelerated::Processor &processor;
    accelerated::Image::Factory &imgFactory;
    accelerated::operations::StandardFactory &opFactory;

    std::shared_ptr<accelerated::Image> visualizationBuffer;
    std::function<void()> visualizationOp;

    std::unique_ptr<CornerResponse> cornerResponse;
    CollectMax collectMax;
    std::vector<KeyPoint> keypoints;

public:
    FeatureDetectorImplementation(
        int w, int h,
        accelerated::Processor &proc,
        accelerated::Image::Factory &images,
        accelerated::operations::StandardFactory &ops,
        const odometry::ParametersTracker &p,
        bool useGpu)
    :
        FeatureDetector(p),
        width(w), height(h),
        cpuProcessor(useGpu ? nullptr : accelerated::Processor::createInstant()),
        processor(useGpu ? proc : *cpuProcessor),
        imgFactory(images),
        opFactory(ops),
        cornerResponse(useGpu
            ? static_cast<CornerResponse*>(new GpuCornerResponse(w, h, images, ops, p))
            : static_cast<CornerResponse*>(new CpuCornerResponse(p))),
        collectMax(width, height, images, ops, p, useGpu)
    {
        log_debug("initialized feature detector (%s)", useGpu ? "GPU" : "CPU-fallback");
    }

    void detect(
        Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) final
    {
        detect(image.getAccImage(), corners, prevCorners, maskRadius).wait();
    }

    accelerated::Future detect(
        accelerated::Image& image,
        std::vector<Feature::Point>& corners,
        const std::vector<Feature::Point>& prevCorners,
        int maskRadius) final
    {
        collectMax(processor, (*cornerResponse)(image), keypoints);
        return processor.enqueue([this, &prevCorners, &corners, maskRadius]() {
            // log_debug("collectMax %p returned %zu keypoints", this, keypoints.size());

            std::stable_sort(keypoints.begin(), keypoints.end(),
                [](const KeyPoint &a, const KeyPoint &b) -> bool {
                    return a.response > b.response;
                });

            corners.clear();
            corners.resize(keypoints.size());
            for (const auto &kp : keypoints) corners.push_back(kp);

            if (maskRadius > 0) applyMinDistance(corners, prevCorners, maskRadius);
            // log_debug("%zu corners left (max %d)", corners.size(), parameters.maxTracks);
        });
    }

    bool supportsAsync() const final {
        return !cpuProcessor;
    }

    void debugVisualize(cv::Mat &target) final {
        if (!visualizationBuffer) {
            visualizationBuffer = imgFactory.create<Ufixed8, 4>(width, height);
            visualizationOp = cornerResponse->debugRenderer(*visualizationBuffer, opFactory);
            //visualizationOp = collectMax.debugRenderer(*visualizationBuffer, opFactory);
        }

        visualizationOp();
        accelerated::opencv::copy(*visualizationBuffer, target);
    }
};
}

FeatureDetector::~FeatureDetector() = default;
FeatureDetector::FeatureDetector(const odometry::ParametersTracker &p)
 : parameters(p) {}

std::unique_ptr<FeatureDetector> FeatureDetector::build(
    int w, int h,
    accelerated::Processor &proc,
    accelerated::Image::Factory &ifac,
    accelerated::operations::StandardFactory &ofac,
    const odometry::ParametersTracker &p) {

    if (p.featureDetector == "GPU-GFTT")
        return std::unique_ptr<FeatureDetector>(
            new FeatureDetectorImplementation(w, h, proc, ifac, ofac, p,
                ifac.getSpec<std::uint8_t, 1>().storageType != accelerated::Image::StorageType::CPU));

    if (p.featureDetector == "FAST")
        return buildLegacyFAST(w, h, p);

    if (p.featureDetector == "GFTT")
        return buildLegacyGFTT(w, h, p);

    log_error("invalid feature detector %s", p.featureDetector.c_str());
    assert(false && "invalid featureDetector");
    return {};
}

}
