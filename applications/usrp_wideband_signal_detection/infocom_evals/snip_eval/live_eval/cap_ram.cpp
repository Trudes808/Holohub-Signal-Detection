// cap_ram — UHD RX capture that receives into a pre-faulted RAM buffer (NO disk I/O during the
// stream) and flushes to a file afterward. This decouples the real-time receive from the slow
// disk write, so 500 MSps cf32 (4 GB/s) captures cleanly even though the NVMe sustains < 1 GB/s.
// Proven-safe design: benchmark_rate shows 0 dropped samples at 500 MSps; the only real-time cost
// here is recv + sc16->fc32 conversion into resident RAM.
//
// usage: cap_ram <addr> <freq_hz> <rate_hz> <gain_db> <duration_s> <channel> <outfile> [bw_hz]
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/metadata.hpp>
#include <complex>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "usage: " << argv[0]
                  << " <addr> <freq_hz> <rate_hz> <gain_db> <duration_s> <channel> <outfile> [bw_hz]\n";
        return 2;
    }
    const std::string addr    = argv[1];
    const double freq         = std::stod(argv[2]);
    const double rate         = std::stod(argv[3]);
    const double gain         = std::stod(argv[4]);
    const double duration     = std::stod(argv[5]);
    const size_t chan         = std::stoul(argv[6]);
    const std::string outfile = argv[7];
    const double bw           = (argc > 8) ? std::stod(argv[8]) : 0.0;

    const std::string dev_args = "addr=" + addr + ",master_clock_rate=500e6";
    std::cerr << "[cap] make device: " << dev_args << "\n";
    auto usrp = uhd::usrp::multi_usrp::make(dev_args);

    usrp->set_rx_rate(rate, chan);
    usrp->set_rx_freq(uhd::tune_request_t(freq), chan);
    usrp->set_rx_gain(gain, chan);
    if (bw > 0.0) usrp->set_rx_bandwidth(bw, chan);

    const double a_rate = usrp->get_rx_rate(chan);
    const double a_freq = usrp->get_rx_freq(chan);
    const double a_gain = usrp->get_rx_gain(chan);
    double a_bw = 0.0; try { a_bw = usrp->get_rx_bandwidth(chan); } catch (...) {}
    std::cerr << "[cap] actual rate=" << a_rate/1e6 << " Msps  freq=" << a_freq/1e6
              << " MHz  gain=" << a_gain << " dB  bw=" << a_bw/1e6 << " MHz\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(750));  // LO lock / AGC settle

    const size_t total = static_cast<size_t>(a_rate * duration + 0.5);
    std::cerr << "[cap] allocating " << (total * sizeof(std::complex<float>) / 1e9)
              << " GB (" << total << " samples) in RAM (zero-init pre-faults pages) ...\n";
    std::vector<std::complex<float>> buff(total);  // value-init touches every page up front
    std::cerr << "[cap] buffer resident, starting stream\n";

    uhd::stream_args_t sa("fc32", "sc16");
    sa.channels = {chan};
    uhd::rx_streamer::sptr rx = usrp->get_rx_stream(sa);
    uhd::stream_cmd_t scmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    scmd.stream_now = true;
    rx->issue_stream_cmd(scmd);

    uhd::rx_metadata_t md;
    size_t got = 0, ovf = 0, errs = 0;
    const size_t CHUNK = 1000000;   // 1M samples/recv
    double timeout = 2.0;
    const auto t0 = std::chrono::steady_clock::now();
    while (got < total) {
        const size_t want = std::min(CHUNK, total - got);
        const size_t n = rx->recv(&buff[got], want, md, timeout);
        timeout = 0.2;
        got += n;
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) { ovf++; continue; }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
            if (errs < 10) std::cerr << "[cap] recv err: " << md.strerror() << "\n";
            errs++;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    rx->issue_stream_cmd(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
    const double rx_sec = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[cap] received " << got << " samples in " << rx_sec << " s ("
              << got/rx_sec/1e6 << " Msps effective), overflows=" << ovf << " errs=" << errs << "\n";

    std::cerr << "[cap] flushing to " << outfile << " ...\n";
    const auto w0 = std::chrono::steady_clock::now();
    std::ofstream f(outfile, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buff.data()),
            static_cast<std::streamsize>(got * sizeof(std::complex<float>)));
    f.close();
    const auto w1 = std::chrono::steady_clock::now();
    std::cerr << "[cap] wrote " << (got * sizeof(std::complex<float>) / 1e9) << " GB in "
              << std::chrono::duration<double>(w1 - w0).count() << " s\n";
    return (ovf > 0 || errs > 0) ? 1 : 0;
}
