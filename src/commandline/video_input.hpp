#ifndef CMDLINE_VIDEO_INPUT_HPP
#define CMDLINE_VIDEO_INPUT_HPP

#include <memory>
#include <string>

namespace odometry { struct ParametersTracker; }
namespace cv { class Mat; }

struct VideoInput {
    static std::unique_ptr<VideoInput> build(
        const std::string &fileName,
        const bool convertVideoToGray = false,
        const bool videoReaderThreads = true,
        const bool ffmpeg = false,
        const std::string &vf = "");
    virtual ~VideoInput();

    virtual double probeFPS() = 0;
    virtual void probeResolution(int &width, int &height) = 0;
    virtual std::shared_ptr<cv::Mat> readFrame() = 0;
    virtual void resize(int width, int height) = 0;
};

#endif
