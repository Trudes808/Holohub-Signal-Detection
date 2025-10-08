// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "CHDR_converter/chdr_rx.h"
#include <fft.hpp>

// #define WRITE_DATA

class UsrpFreqDetectPipeline : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
        if (adv_net_init(adv_net_config) != Status::SUCCESS) {
            HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
            exit(1);
        }
        HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

        auto chdrConverterOp = make_operator<ops::ChdrConverterOpRx>(
            "chdrConverterOp",
            from_config("chdr_converter"));

        auto fftOp = make_operator<ops::FFT>(
            "fftOp",
            from_config("fft"));

        add_operator(chdrConverterOp);
        add_operator(fftOp);
        add_flow(chdrConverterOp, fftOp);

    }
};

int main(int argc, char** argv) {
    holoscan::set_log_pattern("FULL");
    auto app = holoscan::make_application<UsrpFreqDetectPipeline>();

    // Get the configuration file
    if (argc < 1) {
        HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
        return -1;
    }

    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);

    // Check if the file exists
    if (!std::filesystem::exists(config_path)) {
        HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                static_cast<std::string>(config_path));
        return -1;
    }

    // Run
    app->enable_metadata(true);
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));
    app->run();

    shutdown();
    return 0;
}
