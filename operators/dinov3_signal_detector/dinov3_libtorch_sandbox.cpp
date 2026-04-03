// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <torch/script.h>
#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
  std::string script_path;
  std::string mode = "cpu";
  int rows = 250;
  int cols = 20480;
  int input_height = 256;
  int input_width = 512;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    auto next_value = [&](const char* name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++index];
    };

    if (arg == "--script") {
      args.script_path = next_value("--script");
    } else if (arg == "--mode") {
      args.mode = next_value("--mode");
    } else if (arg == "--rows") {
      args.rows = std::stoi(next_value("--rows"));
    } else if (arg == "--cols") {
      args.cols = std::stoi(next_value("--cols"));
    } else if (arg == "--input-height") {
      args.input_height = std::stoi(next_value("--input-height"));
    } else if (arg == "--input-width") {
      args.input_width = std::stoi(next_value("--input-width"));
    } else {
      throw std::runtime_error(std::string("Unknown argument: ") + std::string(arg));
    }
  }

  if (args.script_path.empty()) {
    throw std::runtime_error("--script is required");
  }
  if (args.mode != "cpu" && args.mode != "cuda") {
    throw std::runtime_error("--mode must be 'cpu' or 'cuda'");
  }
  return args;
}

void run_tensor_checks(const Args& args, const std::string& label) {
  std::cout << "[" << label << "] begin\n";

  const size_t total = static_cast<size_t>(args.rows) * static_cast<size_t>(args.cols);
  std::vector<float> host_power_db(total);
  for (size_t index = 0; index < total; ++index) {
    host_power_db[index] = static_cast<float>((index % 1024) * 0.01f - 80.0f);
  }

  auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  auto viewed = torch::from_blob(host_power_db.data(), {args.rows, args.cols}, cpu_options);
  std::cout << "[" << label << "] from_blob cpu ok shape=" << viewed.sizes() << "\n";

  auto cloned = viewed.clone();
  std::cout << "[" << label << "] clone cpu ok shape=" << cloned.sizes() << "\n";

  auto allocated = torch::empty({args.rows, args.cols}, cpu_options);
  std::memcpy(allocated.data_ptr<float>(), host_power_db.data(), total * sizeof(float));
  std::cout << "[" << label << "] empty+memcpy cpu ok shape=" << allocated.sizes() << "\n";

  const bool use_cuda = args.mode == "cuda";
  auto device = use_cuda ? c10::Device(torch::kCUDA, 0) : c10::Device(torch::kCPU);

  if (use_cuda) {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA requested but torch::cuda::is_available() is false");
    }
    auto moved = allocated.to(device);
    std::cout << "[" << label << "] cpu->cuda tensor ok shape=" << moved.sizes() << " device=" << moved.device() << "\n";
  }

  auto module = torch::jit::load(args.script_path, device);
  module.eval();
  std::cout << "[" << label << "] jit load ok mode=" << args.mode << "\n";

  auto input = torch::randn({1, 3, args.input_height, args.input_width},
                            torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto output = module.forward({input});
  if (output.isTensor()) {
    auto tensor = output.toTensor();
    std::cout << "[" << label << "] forward ok shape=" << tensor.sizes() << " device=" << tensor.device() << "\n";
  } else {
    std::cout << "[" << label << "] forward non-tensor output\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);

    std::cout << "torch version: " << TORCH_VERSION << "\n";
    std::cout << "script: " << args.script_path << "\n";
    std::cout << "mode: " << args.mode << "\n";

    run_tensor_checks(args, "main_thread");

    std::exception_ptr worker_error;
    std::thread worker([&]() {
      try {
        run_tensor_checks(args, "worker_thread");
      } catch (...) {
        worker_error = std::current_exception();
      }
    });
    worker.join();
    if (worker_error) {
      std::rethrow_exception(worker_error);
    }

    std::cout << "sandbox complete\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sandbox failed: " << error.what() << "\n";
    return 1;
  }
}