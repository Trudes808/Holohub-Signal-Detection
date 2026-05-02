// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "CHDR_converter/chdr_rx.h"
#include <spectrogram.hpp>
#include <fft.hpp>

// operator that logs information about the processed data
class LogOp: public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LogOp)

    using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

    LogOp() = default;

    void setup(holoscan::OperatorSpec& spec) override {
        auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
        input_port.conditions().emplace_back(
            holoscan::ConditionType::kMessageAvailable,
            std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
        spec.param(num_channels_, "num_channels",
            "Number of Channels",
            "The number of RF channels being processed.", 1);
        spec.param(log_interval_, "log_interval",
            "Log Interval",
            "Interval in seconds to log the data rate statistics.", 5);
        spec.param(log_data_, "log_data",
            "Log Data",
            "If true, log detailed data information for debugging.", false);
    }

    void initialize() {
        holoscan::Operator::initialize();
        total_samples_.resize(num_channels_, 0);
        start_.resize(num_channels_, std::chrono::steady_clock::now());
        elapsed_.resize(num_channels_, std::chrono::steady_clock::duration::zero());
    }

    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override {
        // Receive input tensor and CUDA stream
        auto input = op_input.receive<in_t>("in").value();
        // auto cuda_stream = op_input.receive_cuda_stream("in", true, false);
        auto tensor = std::get<0>(input);
        auto stream = std::get<1>(input);

        // Access metadata
        auto meta = metadata();
        auto channel_num = meta->get<uint16_t>("channel_number", 0);

        // Get timing information
        auto now =  std::chrono::steady_clock::now();
        auto interval = now - start_[channel_num];
        start_[channel_num] = now;

        // Get tensor dimensions
        auto num_samples = tensor.Size(0) * tensor.Size(1);
        auto num_bits = num_samples * sizeof(int16_t) * 2 * 8;  // sc16 = complex int16_t

        // Update statisitics
        total_samples_[channel_num] += num_samples;
        elapsed_[channel_num] += interval;

        // Log statistics
        auto seconds = std::chrono::duration<double>(elapsed_[channel_num]).count();
        if (total_samples_[channel_num] > 0 && seconds >= log_interval_) {
            const double samples_per_second = static_cast<double>(total_samples_[channel_num]) / seconds;
            const double bits_per_second = static_cast<double>(total_samples_[channel_num]) * sizeof(int16_t) * 2 * 8 / seconds;
            HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps)",
                            total_samples_[channel_num],
                            channel_num,
                            samples_per_second / 1e6,
                            bits_per_second / 1e9);
            total_samples_[channel_num] = 0;
            elapsed_[channel_num] = std::chrono::steady_clock::duration::zero();
        }

        // Log data for debugging
        if (log_data_) {
            HOLOSCAN_LOG_INFO("Received tensor from channel {} with rank {} and shape: ({}, {})",
                channel_num, tensor.Rank(), tensor.Size(0), tensor.Size(1));
            make_tensor(samples_, tensor.Shape(), MATX_HOST_MEMORY);
            auto result = cudaMemcpy(
                samples_.Data(),
                tensor.Data(),
                tensor.Size(0) * tensor.Size(1) * sizeof(complex),
                cudaMemcpyDeviceToHost
            );
            if (result != cudaSuccess) {
                HOLOSCAN_LOG_ERROR("cudaMemcpy failed with error code: {}", static_cast<int>(result));
            }
            else {
                cudaDeviceSynchronize();
                HOLOSCAN_LOG_INFO("First 1024 FFT samples with stride 20 of channel {}:", channel_num);
                set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
                print(slice<1>(samples_, {0, 0}, {matxDropDim, matxEnd}, {1, 20}));
            }
            // Below code not working when called after FFT operator
            // set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
            // print(slice<1>(tensor, {0, 0}, {matxDropDim, 1024}));
        }
    }
 private:
    holoscan::Parameter<int> num_channels_;
    holoscan::Parameter<int> log_interval_;
    holoscan::Parameter<bool> log_data_;
    std::vector<int64_t> total_samples_;
    std::vector<std::chrono::steady_clock::time_point> start_;
    std::vector<std::chrono::steady_clock::duration> elapsed_;
    tensor_t<complex, 2> samples_;
};

class UsrpFreqDetectPipeline : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        const bool enable_spectrogram = from_config("pipeline.enable_spectrogram").as<bool>();
        const bool enable_logger = from_config("pipeline.enable_logger").as<bool>();

        auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
        if (adv_net_init(adv_net_config) != Status::SUCCESS) {
            HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
            exit(1);
        }
        HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

        auto chdrConverterOp = make_operator<ops::ChdrConverterOpRx>(
            "chdrConverterOp",
            from_config("chdr_converter"));
        const int pipeline_channels = std::max(1, from_config("chdr_converter.num_channels").as<int>());

        std::vector<std::shared_ptr<holoscan::Operator>> fftOps;
        std::vector<std::shared_ptr<holoscan::Operator>> spectrogramOps(static_cast<size_t>(pipeline_channels));
        std::vector<std::shared_ptr<holoscan::Operator>> logOps(static_cast<size_t>(pipeline_channels));
        fftOps.reserve(static_cast<size_t>(pipeline_channels));

        for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
            fftOps.push_back(make_operator<ops::FFT>(
                std::string("fftOpCh") + std::to_string(channel_index),
                from_config("fft")));
            if (enable_spectrogram) {
                spectrogramOps[static_cast<size_t>(channel_index)] = make_operator<ops::Spectrogram>(
                    std::string("spectrogramOpCh") + std::to_string(channel_index),
                    from_config("spectrogram"));
            }
            if (enable_logger) {
                logOps[static_cast<size_t>(channel_index)] = make_operator<LogOp>(
                    std::string("logOpCh") + std::to_string(channel_index),
                    from_config("logger"),
                    make_condition<CountCondition>(from_config("num_runs").as<int64_t>()));
            }
        }

        add_operator(chdrConverterOp);
        for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
            add_operator(fftOps[static_cast<size_t>(channel_index)]);
            if (enable_spectrogram) {
                add_operator(spectrogramOps[static_cast<size_t>(channel_index)]);
            }
            if (enable_logger) {
                add_operator(logOps[static_cast<size_t>(channel_index)]);
            }

            const char* chdr_port = channel_index == 0 ? "out0" : "out1";
            add_flow(chdrConverterOp, fftOps[static_cast<size_t>(channel_index)], {{chdr_port, "in"}});
            if (enable_spectrogram) {
                add_flow(fftOps[static_cast<size_t>(channel_index)], spectrogramOps[static_cast<size_t>(channel_index)]);
            }
            if (enable_logger) {
                add_flow(fftOps[static_cast<size_t>(channel_index)], logOps[static_cast<size_t>(channel_index)]);
            }
        }
    }
};

int main(int argc, char** argv) {
    // Create the application
    auto app = holoscan::make_application<UsrpFreqDetectPipeline>();
    
    // Enable metadata for all operators
    app->enable_metadata(true);

    // Check for required configuration file argument
    if (argc < 2) {
        HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
        return -1;
    }

    // Get the full path to the configuration file
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);

    // Check if the configuration file exists
    if (!std::filesystem::exists(config_path)) {
        HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                static_cast<std::string>(config_path));
        return -1;
    }

    // Apply configuration from file
    app->config(config_path);

    // Configure the event-based scheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));

    // Run the application
    app->run();

    // Shutdown
    shutdown();
    return 0;
}
