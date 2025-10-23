// SPDX-FileCopyrightText: 2025 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "CHDR_converter/chdr_rx.h"
#include <fft.hpp>

// operator that logs information about the processed data
class LogOp: public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LogOp)

    using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

    LogOp() = default;

    void setup(holoscan::OperatorSpec& spec) override {
        spec.input<in_t>("in");
    }

    void initialize() {
        holoscan::Operator::initialize();
        total_samples_ = 0;
        start_ = std::chrono::steady_clock::now();
        elapsed_ = std::chrono::steady_clock::duration::zero();
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
        auto interval = now - start_;
        start_ = now;

        // Get tensor dimensions
        auto num_samples = tensor.Size(0) * tensor.Size(1);
        auto num_bits = num_samples * sizeof(int16_t) * 2 * 8;  // sc16 = complex int16_t

        // Update statisitics
        total_samples_ += num_samples;
        elapsed_ += interval;

        // Log statistics
        auto seconds = std::chrono::duration<double>(elapsed_).count();
        if (total_samples_ > 0 && seconds >= LOG_INTERVAL) {
            auto duration = std::chrono::duration<double>(interval).count();
            HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps)",
                            total_samples_,
                            channel_num,
                            num_samples / duration / 1e6,
                            num_bits / duration / 1e9);
            total_samples_ = 0;
            elapsed_ = std::chrono::steady_clock::duration::zero();
        }

        // Debugging
        if (false) {
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
        // Debugging end
    }
 private:
    static constexpr int LOG_INTERVAL = 1;  // log interval in seconds
    int64_t total_samples_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::duration elapsed_;
    tensor_t<complex, 2> samples_;
};

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

        auto logOp = make_operator<LogOp>(
            "logOp",
            make_condition<CountCondition>(from_config("num_runs").as<int64_t>())
            // make_condition<CudaStreamCondition>("stream_sync", Arg("receiver", "in"))
        );

        add_operator(chdrConverterOp);
        add_operator(fftOp);
        add_operator(logOp);
        add_flow(chdrConverterOp, fftOp);
        add_flow(fftOp, logOp);
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
