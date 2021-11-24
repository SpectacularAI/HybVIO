#ifndef TRACKER_VIDEOUTIL_H_
#define TRACKER_VIDEOUTIL_H_

#include <string>

namespace videoutil {

double ffprobeFps(const std::string &videoPath);
bool ffprobeResolution(const std::string &videoPath, int &width, int &height);
void changeVideoFps(const std::string &src, const std::string &dst, double fps);

}

#endif
