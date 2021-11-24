#include "video_input.hpp"
#include "../util/allocator.hpp"
#include "../util/logging.hpp"
#include "../odometry/parameters.hpp"
#include "videoutil.hpp"
#include "../util/bounded_processing_queue.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

namespace {

struct Reader {
    virtual ~Reader() = default;
    virtual bool read(cv::Mat &target) = 0;
    virtual bool isOk() const = 0;
};

class VideoInputImplementation : public VideoInput {
private:
    std::string videoPath;
    std::unique_ptr<Reader> reader;
    cv::Mat resizeSource, colorSource;
    int resizeWidth = -1, resizeHeight = -1;

    static constexpr size_t BUFFER_SIZE = 4;
    util::BoundedInputQueue<cv::Mat> queue;

public:
    VideoInputImplementation(const std::string &videoPath, std::unique_ptr<Reader> reader, bool ownThread, bool convertToGray) :
        videoPath(videoPath),
        reader(std::move(reader)),
        queue(ownThread ? BUFFER_SIZE : 0, [this, convertToGray](cv::Mat &frame) -> bool {
            const bool resized = resizeWidth > 0;
            cv::Mat &target = convertToGray ? colorSource : (resized ? resizeSource : frame);
            if (this->reader->read(target)) {
                if (convertToGray) {
                    cv::Mat &convertTarget = resized ? resizeSource : frame;
                    switch (target.channels()) {
                        case 1: target.copyTo(convertTarget); break;
                        case 3: cv::cvtColor(target, convertTarget, cv::COLOR_BGR2GRAY); break;
                        case 4: cv::cvtColor(target, convertTarget, cv::COLOR_BGRA2GRAY); break;
                        default: assert(false && "invalid color format"); break;
                    }
                }
                if (resized) {
                    cv::resize(resizeSource, frame, cv::Size(resizeWidth, resizeHeight), 0, 0, cv::INTER_CUBIC);
                }
            } else {
                return false;
            }
            return true;
        })
    {}

    double probeFPS() final {
        return videoutil::ffprobeFps(videoPath);
    }

    void probeResolution(int &w, int &h) final {
        bool success = videoutil::ffprobeResolution(videoPath, w, h);
        assert(success && w > 0 && h > 0);
    }

    std::shared_ptr<cv::Mat> readFrame() final {
        return queue.get();
    }

    void resize(int width, int height) final {
        resizeWidth = width;
        resizeHeight = height;
    }
};

struct FFMpegReader: Reader {
    int width = 0;
    int height = 0;
    FILE *pipe = nullptr;

    FFMpegReader(const std::string &videoPath, const std::string vf) {
        bool success = videoutil::ffprobeResolution(videoPath, width, height);
        if (success) {
            assert(success && width > 0 && height > 0);
            std::stringstream filters;
            if (!vf.empty()) {
                filters << " -vf \"" << vf << "\"";
            }

            constexpr bool VERBOSE = false;

            std::stringstream ss; ss << "ffmpeg -i "
                << videoPath
                << " -f rawvideo -vcodec rawvideo -vsync vfr"
                << filters.str()
                << " -pix_fmt bgr24 -"
                << (VERBOSE ? "" : " 2>/dev/null");

            log_debug("Running: %s", ss.str().c_str());
            pipe = popen(ss.str().c_str(), "r");
        }
    }

    ~FFMpegReader() {
        fflush(pipe);
        pclose(pipe);
    }

    bool read(cv::Mat &frame) final {

        if (frame.empty()) {
            frame = cv::Mat(height, width, CV_8UC3);
        }
        assert(frame.rows == height && frame.cols == width);
        assert(frame.type() == CV_8UC3);
        int n = height * width * 3;
        assert(pipe);
        assert(frame.data);
        int count = fread(frame.data, 1, n, pipe);

        return count == n;
    }

    bool isOk() const final {
        return pipe != nullptr;
    }
};

struct OpenCVReader : Reader {
    cv::VideoCapture videoCapture;

    OpenCVReader(const std::string &videoPath) {
        videoCapture = cv::VideoCapture(videoPath);
    }

    bool read(cv::Mat &frame) final {
        return videoCapture.read(frame);
    }

    bool isOk() const final {
        return videoCapture.isOpened();
    }
};
}

std::unique_ptr<VideoInput> VideoInput::build(
        const std::string &fileName,
        const bool convertVideoToGray,
        const bool videoReaderThreads,
        const bool ffmpeg,
        const std::string &vf) {
    auto reader = ffmpeg
        ? std::unique_ptr<Reader>(new FFMpegReader(fileName, vf))
        : std::unique_ptr<Reader>(new OpenCVReader(fileName));
    if (!reader->isOk()) {
        std::cout << "Couldn't open video " << fileName << std::endl;
        return nullptr;
    }
    return std::unique_ptr<VideoInput>(new VideoInputImplementation(fileName,
        std::move(reader),
        videoReaderThreads,
        convertVideoToGray));
}

VideoInput::~VideoInput() = default;
