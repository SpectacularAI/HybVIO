#ifndef IMU_VISUALIZATION_HPP
#define IMU_VISUALIZATION_HPP

#include "../api/types.hpp"
#include <opencv2/opencv.hpp>

class ImuVisualization {
private:
    struct Sample {
        double t;
        api::Vector3d v;
    };

    static constexpr int W = 800;
    static constexpr int H = 400;
    static constexpr int gray = 0xaa;

    std::vector<Sample> buffer;
    cv::Scalar bgColor;
    cv::Mat canvas;
    double scale;

public:
    ImuVisualization(double scale) :
        buffer(),
        bgColor(cv::Scalar(gray, gray, gray, 0xff)),
        canvas(cv::Mat(H, W, CV_8UC4, bgColor)),
        scale(scale)
    {}

    void addSample(double t, const api::Vector3d &value) {
        buffer.push_back({ t, value });
        constexpr std::size_t MAX_IMU_BUFFER_SIZE = 2000;
        while (buffer.size() > MAX_IMU_BUFFER_SIZE)
            buffer.erase(buffer.begin());
    }

    const cv::Mat &draw(double t) {
        canvas = bgColor;

        constexpr double T_WINDOW_BACK = 5.0;
        constexpr double T_WINDOW_FRONT = 0.5;
        const double V_WINDOW = scale;

        const double t0 = t - T_WINDOW_BACK;
        const double t1 = t + T_WINDOW_FRONT;
        const double v0 = -V_WINDOW*0.5;
        const double v1 = V_WINDOW*0.5;

        const auto tToPixels = [&](double t) -> int {
            return static_cast<int>((t - t0) / (t1 - t0) * canvas.cols);
        };

        const auto vToPixels = [&](double v) -> int {
            return static_cast<int>((1.0 - (v - v0) / (v1 - v0)) * canvas.rows);
        };

        // current time as a black line
        cv::line(canvas,
            cv::Point2f(tToPixels(t), vToPixels(v0)),
            cv::Point2f(tToPixels(t), vToPixels(v1)),
            cv::Scalar(0, 0, 0, 0xff));

        // t-axis
        cv::line(canvas,
            cv::Point2f(tToPixels(t0), vToPixels(0)),
            cv::Point2f(tToPixels(t1), vToPixels(0)),
            cv::Scalar(0, 0, 0, 0xff));

        const cv::Scalar xyzColors[] = {
            cv::Scalar(0, 0, 0xff, 0xff), // red
            cv::Scalar(0, 0xff, 0, 0xff), // green
            cv::Scalar(0xff, 0, 0, 0xff) // blue (OpenCV uses BGR colors)
        };

        const Sample *prev = nullptr;
        for (const auto &sample : buffer) {
            if (prev != nullptr) {
                const double prevVals[] = { prev->v.x, prev->v.y, prev->v.z };
                const double vals[] = { sample.v.x, sample.v.y, sample.v.z };
                const int prevT = tToPixels(prev->t);
                const int curT = tToPixels(sample.t);
                for (int i = 0; i < 3; ++i) {
                    cv::line(canvas,
                        cv::Point2f(prevT, vToPixels(prevVals[i])),
                        cv::Point2f(curT, vToPixels(vals[i])),
                        xyzColors[i]);
                }
            }
            prev = &sample;
        }

        return canvas;
    }
};

#endif
