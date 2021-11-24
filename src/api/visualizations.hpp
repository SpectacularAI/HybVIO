#ifndef VISUALIZATIONS_HPP
#define VISUALIZATIONS_HPP

#include "vio.hpp"
#include "internal.hpp"

namespace accelerated { struct Image; }

namespace api {

class VisualizationImplementationBase : public Visualization {

public:
    std::unique_ptr<accelerated::Image> createDefaultRenderTarget() override {
        assert(false && "Not implemented");
    }

    void update(std::shared_ptr<const VioApi::VioOutput> output) override {
        (void)output;
        assert(false && "Not implemented");
    }

    void render(cv::Mat &output) override {
        (void)output;
        assert(false && "Not implemented");
    }

    void render(accelerated::Image &target) override {
        (void)target;
        assert(false && "Not implemented");
    };

    void render() override {
        assert(false && "Not implemented");
    };

    bool ready() const override {
        return true; // most visualizations are always ready
    }
};


class VisualizationVideoOutput : public VisualizationImplementationBase {
    std::shared_ptr<const VioApi::VioOutput> currentOutput;
    InternalAPI* api;
public:
    VisualizationVideoOutput(InternalAPI* api) : api(api) {}
    // This depends on lazy init that happens after processing first frame, not a very good practice
    std::unique_ptr<accelerated::Image> createDefaultRenderTarget() final;
    void update(std::shared_ptr<const VioApi::VioOutput> output) final;
    void render(cv::Mat &output) final;
    void render(accelerated::Image &target) final;
};

class VisualizationKfCorrelation : public VisualizationImplementationBase {
    InternalAPI* api;
public:
    VisualizationKfCorrelation(InternalAPI* api) : api(api) {}
    void render(cv::Mat &output) final;
};

class VisualizationPose : public VisualizationImplementationBase {
    std::shared_ptr<const VioApi::VioOutput> currentOutput;
    InternalAPI* api;
public:
    VisualizationPose(InternalAPI* api) : api(api) {}
    void update(std::shared_ptr<const VioApi::VioOutput> output) final;
    void render(cv::Mat &target) final;
    bool ready() const final;
};

class VisualizationCovarianceMagnitudes : public VisualizationImplementationBase {
    InternalAPI* api;
public:
    VisualizationCovarianceMagnitudes(InternalAPI* api) : api(api) {}
    void render(cv::Mat &target) final;
};

} // namespace api

#endif // VISUALIZATIONS_HPP
