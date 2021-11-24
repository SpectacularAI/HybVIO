#include "videoutil.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <iostream>
#include <sstream>

namespace {
// Run a shell command and return its stdout (not stderr).
std::string exec(const std::string &cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}
}

namespace videoutil {

// Use ffmpeg (likely installed as dependency of OpenCV) to get video frame rate.
double ffprobeFps(const std::string &videoPath) {
    // `r_frame_rate` has been confirmed to sometimes give wrong results compared to `avg_frame_rate` on Pixel and iPhone 11.
    std::string cmd = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=avg_frame_rate ";

    // Discard stderr (e.g. the shell's "ffprobe: command not found" print).
    std::string fpsText = exec(cmd + videoPath + " 2>/dev/null");

    int a = 0, b = 0;
    if (sscanf(fpsText.c_str(), "%d/%d", &a, &b) == 2) {
        return static_cast<double>(a) / static_cast<double>(b);
    }
    else {
        std::cout << "Failed to extract fps information from video. fpsText: " << fpsText << std::endl;;
        return 0.0;
    }
}

// Get video resolution.
bool ffprobeResolution(const std::string &videoPath, int &width, int &height) {
    std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 " + videoPath;
    std::string resolutionText = exec(cmd + " 2>/dev/null");
    if (sscanf(resolutionText.c_str(), "%dx%d", &width, &height) == 2) {
        return true;
    }
    else {
        std::cout << "Failed to extract resolution information from video. resolutionText: " << resolutionText << std::endl;
        return false;
    }
}

// Change video fps without re-encoding.
void changeVideoFps(const std::string &src, const std::string &dst, double fps) {
    // Probably need to tune the commands if this is used with anything other than h264 and mp4.
    std::string tmp = "tmp.h264";
    std::stringstream cmd1;
    cmd1 << "ffmpeg -y -i " << src << " -c copy -f h264 " << tmp;
    exec(cmd1.str());

    std::stringstream cmd2;
    cmd2 << "ffmpeg -y -r " << fps << " -i " << tmp << " -c copy " << dst;
    exec(cmd2.str());

    std::remove(tmp.c_str());
}

}
