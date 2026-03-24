
================================================================================
STEP 1 -- PREREQUISITES (run once on host, and after every reboot)
================================================================================

# Load GPUDirect
nv_driver_version=$(nvidia-smi -q | grep 'Driver Version' | awk '{print $4}' | cut -d. -f1)
sudo dpkg-reconfigure nvidia-dkms-$nv_driver_version
sudo modprobe nvidia-peermem
lsmod | grep nvidia_peermem        # verify it shows nvidia_peermem loaded

# Hugepages (run after every reboot)
sudo sysctl -w vm.nr_hugepages=8
cat /proc/meminfo | grep HugePages_Total    # verify shows 8

# Mount hugepages (run after every reboot)
sudo mkdir -p /dev/hugepages
sudo mount -t hugetlbfs nodev /dev/hugepages

================================================================================
STEP 2 -- CLONE THE REPO (run once)
================================================================================

cd ~
git clone --branch users/jlange/usrp_freq_detection \
  https://github.com/mbr0wn/holohub-dev.git
cd holohub-dev

================================================================================
STEP 3 -- BUILD THE CONTAINER (run once, takes 15-20 minutes)
================================================================================

cd ~/holohub-dev
sudo ./holohub build-container usrp_freq_detection

# Verify
docker images | grep holohub
# Should show: holohub:usrp_freq_detection

================================================================================
STEP 3B -- REBUILD + RESTART THE CONTAINER (after code changes)
================================================================================

cd ~/holohub-dev
./applications/usrp_freq_detection/restart_usrp_container.sh

================================================================================
STEP 4 -- LAUNCH THE CONTAINER
================================================================================

sudo docker run --name holohub_usrp_freq_detection --privileged --net host --interactive --tty \
  -u 0:0 \
  -v ~/holohub-dev:/workspace/holohub \
  -v /dev/hugepages:/dev/hugepages \
  -w /workspace/holohub \
  --runtime nvidia \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  holohub:usrp_freq_detection

# You are now inside the container. Prompt will show:
# root@ecen-a58853:/workspace/holohub#

================================================================================
STEP 5 -- BUILD INSIDE CONTAINER (run once)
================================================================================

# DPDK struct compatibility fix is already integrated in this branch.
# Build + verify in one command:
/workspace/holohub/applications/usrp_freq_detection/bootstrap_build.sh

# Verify binary exists
ls /workspace/holohub/build/usrp_freq_detection/applications/usrp_freq_detection/usrp_freq_detection

================================================================================
STEP 6 -- DEFAULT CONFIG FOR SINGLE CHANNEL + 491.52 MSPS
================================================================================

# These values are now defaults in source config.yaml:
#   address: 0000:a2:00.0
#   CPU num_bufs: 51200
#   GPU num_bufs: 25000
#   batch_size: 5000
#   num_ffts_per_batch: 250
#   num_bursts: 250
#
# If your NIC PCIe address is different, edit the source config here:
#   ~/holohub-dev/applications/usrp_freq_detection/config.yaml

# Verify
grep -E "address:|num_bufs|batch_size|num_ffts|num_bursts" \
  ~/holohub-dev/applications/usrp_freq_detection/config.yaml

================================================================================
STEP 7 -- RUN THE PIPELINE (Terminal 1, inside container)
================================================================================

cd /workspace/holohub/build/usrp_freq_detection/applications/usrp_freq_detection/
./usrp_freq_detection config.yaml

# Wait until you see:
# [info] Waiting for completion...
# Then start the X410 stream in Terminal 2

================================================================================
STEP 8 -- START X410 STREAMING (Terminal 2, outside container)
================================================================================

cd ~/holohub-dev/applications/usrp_freq_detection

python3 rx_to_remote_udp.py \
  --args "addr=192.168.10.2" \
  --freq 1e9 \
  --rate 491.52e6 \
  --gain 30 \
  --channels 0 \
  --dest-addr 192.168.100.51 \
  --dest-port 1234 \
  --keep-hdr \
  --adapter sfp1 \
  --dest-mac-addr E0:9D:73:E0:5B:E6

# To change center frequency, modify --freq (in Hz)
# Example: 2.4 GHz = --freq 2.4e9
# Example: 915 MHz = --freq 915e6

# To change gain, modify --gain (0-60 dB)

================================================================================
EXPECTED OUTPUT
================================================================================

Pipeline terminal should show:
  [info] Processed 256000000 samples from channel 0 at ~253 MSps (8.1 Gbps)
  [info] Processed 256000000 samples from channel 0 at ~252 MSps (8.1 Gbps)
  ...

Stop with Ctrl+C on both terminals.

================================================================================
KNOWN ISSUES AND FIXES
================================================================================

Issue: hugepages not mounted
Fix  : sudo mount -t hugetlbfs nodev /dev/hugepages

Issue: nvidia_peermem not loaded
Fix  : sudo modprobe nvidia-peermem

Issue: DMA map error (GPU BAR1 limit)
Fix  : Keep CH1_Data_RX_GPU and CH2_Data_RX_GPU num_bufs <= 25000

Issue: DPDK rte_flow_field_data compile error
Fix  : Already integrated in this branch. Pull latest holohub-sage-dev.

Issue: Config file not found when running binary
Fix  : Run binary from the same directory as config.yaml
       cd /workspace/holohub/build/usrp_freq_detection/applications/usrp_freq_detection/
       ./usrp_freq_detection config.yaml

================================================================================
KEY FILES
================================================================================

Source config    : ~/holohub-dev/applications/usrp_freq_detection/config.yaml
Build config     : ~/holohub-dev/build/usrp_freq_detection/applications/usrp_freq_detection/config.yaml
Pipeline binary  : ~/holohub-dev/build/usrp_freq_detection/applications/usrp_freq_detection/usrp_freq_detection
Streaming script : ~/holohub-dev/applications/usrp_freq_detection/rx_to_remote_udp.py
Dockerfile       : ~/holohub-dev/applications/usrp_freq_detection/Dockerfile
Main source      : ~/holohub-dev/applications/usrp_freq_detection/main.cpp
Build helper     : ~/holohub-dev/applications/usrp_freq_detection/bootstrap_build.sh
Restart helper   : ~/holohub-dev/applications/usrp_freq_detection/restart_usrp_container.sh


================================================================================
END OF README
================================================================================
