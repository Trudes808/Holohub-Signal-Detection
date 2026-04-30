# USRP-Holoscan FFT Demo (preliminary -- not public)


Follow these steps to run this application:


- Use the latest (`master` branch!) version of UHD directly from GitHub, the
  latest stable version (4.9.0) does not yet have all the required features.
- Make sure to also get a compatible FPGA image. For X410, a `CG_400` image is
  required for 100 GbE and the full 500 Msps.
- The images can be downloaded (after installing the latest UHD) by running
  `uhd_images_downloader -t x410 -t fpga`. They can then be flashed onto the
  USRP using `uhd_image_loader -a type=x4xx,addr=$IP_ADDRESS,fpga=CG_400`.
- Connect the USRP both with a 1 GbE Ethernet connetion (to the RJ45) and with
  100 GbE between the computer with the GPU and USRP's QSFP port.
- Set the MTU on the ConnectX NIC to 9000.
- Make sure you can reach the USRP with `uhd_usrp_probe -a addr=$RJ45_ADDRESS`
  and also with `uhd_usrp_probe -a addr=$QSFP_IP_ADDRESS`.
- We recommend using the standard 192.168.10.2 addresses.

If everything is working, start Holoscan from the top level of this repo:

    ./holohub run usrp_freq_detection --docker-opts "-u root --privileged -v /mnt/huge:/mnt/huge"

  To save spectrogram debug images to the host, also mount an output directory and
  make sure it matches `spectrogram.output_dir` in
  `applications/usrp_freq_detection/config.yaml` (default: `/workspace/spectrograms`):

    mkdir -p /tmp/usrp_spectrograms
    ./holohub run usrp_freq_detection --docker-opts "-u root --privileged -v /mnt/huge:/mnt/huge -v /tmp/usrp_spectrograms:/workspace/spectrograms"


Then, start the raw UDP stream using the patched version of rx_to_remote_udp.py
that is inside the `applications/usrp_freq_detection` directory. You need to
provide the RJ45 address in the first argument, and the remote computer's 100 GbE
IP address in the `-i` argument. You may also need to specify the MAC address
of the ConnectX NIC's port (this is something that will be fixed). Don't forget
the `spp` and `mtu` arguments. This command will work:


    python3 ./applications/usrp_freq_detection/rx_to_remote_udp.py \
        -a addr=$RJ45_IP_ADDRESS,master_clock_rate=500e6 -f 1e9 \
        -i $REMOTE_IP_ADDRESS -p 1234 --adapter sfp0 \
        --dest-mac-addr $REMOTE_MAC_ADDR --mtu 8000 --spp 1024

  To tune different center frequencies per channel, provide one `-f/--freq`
  value per selected channel, for example `-c 0 1 -f 2.40e9 2.45e9 -p 1234 1235`.

  The Holoscan receive pipeline already exports per-message operator metadata.
  `ChdrConverterOpRx` always sets `channel_number` and `chdr_emit_ts_ns`, and it
  can now also attach `rx_center_frequency_hz` and `rx_sample_rate_hz`. Because
  the sender script and the Holoscan app are separate processes, the app cannot
  discover those RF settings automatically; set
  `chdr_converter.channel_center_frequencies_hz` and
  `chdr_converter.channel_sample_rates_hz` in
  `applications/usrp_freq_detection/config.yaml` if downstream operators need
  those values.


The Holoscan app should run and report the streaming throughput rate.
