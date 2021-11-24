#ifndef VIO_API_HPP
#define VIO_API_HPP

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <istream>

#include "types.hpp"

// forward declarations for visualizations
namespace cv { class Mat; }
namespace accelerated { struct Image; }

namespace api {

struct Visualization;

class VioApi {
public:
    virtual ~VioApi() = default;

    enum class ColorFormat {
        GRAY,
        RGB,
        RGBA,
        RGBA_EXTERNAL_OES
    };

    struct VioOutput {
        /**
         * Current tracking status
         */
        TrackingStatus status;

        /**
         * The current pose, with the timestamp in the clock used for inputted
         * ensor data and camera frames.
         */
        Pose pose;

        /**
         * Velocity vector (xyz) in m/s in the coordinate system used by pose.
         */
        Vector3d velocity;

        /**
         * Uncertainty of the current position as a 3x3 covariance matrix
         */
        Matrix3d positionCovariance;

        /**
         * List of poses, where the first element corresponds to
         * the value returned by getPose and the following (zero or more)
         * values are the recent smoothed historical positions
         */
        std::vector<Pose> poseTrail;

        /**
         * Point cloud (list of FeaturePoints) that correspond to
         * features currently seen by the camera.
         */
        std::vector<FeaturePoint> pointCloud;

        /**
         * The input frame tag. This is the value given in addFrame... methods
         */
        int tag;

        /**
         * Returns the output to a JSON string if supported. Otheriwse returns
         * an empty string.
         */
        virtual std::string asJson() const { return ""; };

        virtual ~VioOutput() = default;
    };

    // Output API, called when ever there is a new output available
    std::function<void(std::shared_ptr<const VioOutput>)> onOutput;

    // Thread-safe input API. These methods can be called from any thread

    virtual void addAcc(double t, const Vector3d &sample) = 0;

    virtual void addGyro(double t, const Vector3d &sample) = 0;

    virtual void addFrameMono(double t, int w, int h,
        const std::uint8_t *data,
        ColorFormat colorFormat, int tag = 0) = 0;

    virtual void addFrameStereo(double t, int w, int h,
        const std::uint8_t *data0, const std::uint8_t *data1,
        ColorFormat colorFormat, int tag = 0) = 0;

    // This could also be detached from the main API and put into its own file
    // so that visualization can only access Output but not any algorithm
    // internals.
    virtual std::shared_ptr<Visualization> createVisualization(const std::string &type) = 0;
};

// Visualizations
struct Visualization {
    /**
     * If supported creates a default image with correct dimensions and type supported by Visualization.
     */
    virtual std::unique_ptr<accelerated::Image> createDefaultRenderTarget() = 0;

    virtual void update(std::shared_ptr<const VioApi::VioOutput> output) = 0;
    virtual void render(cv::Mat &target) = 0;
    virtual void render(accelerated::Image &target) = 0;
    virtual void render() = 0;

    // Check if the visualization is ready to be rendered
    virtual bool ready() const = 0;

    virtual ~Visualization() = default;
};

std::unique_ptr<api::VioApi> buildVio(std::istream &calibrationJson, std::istream &configYaml);

} // namespace api

#endif // VIO_API_HPP
