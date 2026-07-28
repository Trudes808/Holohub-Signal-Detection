function out = decode_waveforms_24576(target, varargin)
%DECODE_WAVEFORMS_24576 Decode 245.76 MSps waveforms: BER (digital) / audio (FM).
%
% Reads the metadata stored beside each waveform, rationally resamples the
% 245.76 MSps IQ back to its native sample rate, runs the matching receive
% chain, and returns BER (digital classes) or recovered audio + audio quality
% (the two FM classes).
%
% SINGLE waveform:
%   r = decode_waveforms_24576("generated_waveforms_24576/QPSK/<name>.mat")
%   r = decode_waveforms_24576(path, "Channel","wireless")           % add a channel
%   r = decode_waveforms_24576(path, "SNRdB",30, "FrequencyOffsetHz",1e3)
%   r = decode_waveforms_24576(rx, "Fs",245.76e6, "Metadata",md, "TxBits",bits)
%
% BATCH (directory or manifest .csv): decode every waveform, write a results CSV:
%   t = decode_waveforms_24576("generated_waveforms_24576")                 % clean -> BER 0
%   t = decode_waveforms_24576("generated_waveforms_24576","Channel","wireless")
%
% Channel name-value args (applied at 245.76 MSps before the receive chain):
%   Channel ("none"|"wireless"), SNRdB, Gain, PhaseOffsetRad,
%   FrequencyOffsetHz, MultipathDelays (samples), MultipathGainsDb, RandomSeed.

p = inputParser;
p.addRequired("target");
p.addParameter("Fs", []);
p.addParameter("Metadata", []);
p.addParameter("TxBits", []);
p.addParameter("Channel", "none");
p.addParameter("SNRdB", Inf);
p.addParameter("Gain", 1);
p.addParameter("PhaseOffsetRad", 0);
p.addParameter("FrequencyOffsetHz", 0);
p.addParameter("MultipathDelays", []);
p.addParameter("MultipathGainsDb", []);
p.addParameter("RandomSeed", 1234);
p.addParameter("OutputAudioFile", "");
p.addParameter("AudioCompareRateHz", 48000);
p.parse(target, varargin{:});
opts = p.Results;
opts.Channel = string(opts.Channel);

% ---- batch mode (directory or manifest csv) ----
if (ischar(target) || isstring(target)) && isBatchTarget(target)
    out = batchDecode(string(target), opts);
    return;
end

out = decodeOne(opts);
end

% ======================================================================= %
%                          SINGLE-WAVEFORM DECODE                         %
% ======================================================================= %
function result = decodeOne(opts)
[rx, Fs, metadata, txBits, sourcePath] = loadDecodeInput(opts);
rx = applyChannel(rx, Fs, opts, metadata);

standard = string(metadata.standard);
isFM = ismember(standard, ["MP3 audio-source narrowband FM","MP3 audio-source broadband FM"]);

if isFM
    result = receiveAudioFM(rx, metadata, opts, sourcePath);
    return;
end

switch standard
    case "5G NR downlink"
        [decodedBits, referenceBits] = decode5GDownlink(rx, metadata, txBits);
    case "IEEE 802.11ax HE-SU"
        [decodedBits, referenceBits] = decodeWLAN(rx, metadata, txBits);
    case "Generic single-carrier"
        [decodedBits, referenceBits] = decodeSingleCarrier(rx, metadata, txBits);
    case "Generic OFDM"
        [decodedBits, referenceBits] = decodeOFDM(rx, metadata, txBits);
    case "Bluetooth BR/EDR"
        decodedBits = decodeBluetoothBREDR(rx, metadata); referenceBits = txBits;
    case "Bluetooth LE"
        decodedBits = decodeBluetoothLE(rx, metadata);   referenceBits = txBits;
    otherwise
        error("decode_waveforms_24576:Unsupported", ...
            "No decoder for standard '%s' (%s).", standard, string(metadata.waveformName));
end

referenceBits = logical(referenceBits(:));
decodedBits = logical(decodedBits(:));
n = min(numel(referenceBits), numel(decodedBits));
if n == 0
    error("decode_waveforms_24576:NoBits", "No decoded bits for %s.", string(metadata.waveformName));
end
bitErrors = sum(xor(referenceBits(1:n), decodedBits(1:n)));

result = struct;
result.waveformName = string(metadata.waveformName);
result.class = string(metadata.class);
result.standard = standard;
result.sourcePath = string(sourcePath);
result.isFM = false;
result.numComparedBits = n;
result.bitErrors = bitErrors;
result.BER = bitErrors / n;
result.audioSNRdB = NaN;
result.audioCorr = NaN;
result.channel = channelSummary(opts);
end

% ======================================================================= %
%                       5G NR DOWNLINK RECEIVER                           %
% ======================================================================= %
function [decodedBits, referenceBits] = decode5GDownlink(rx, metadata, txBits)
cfg = metadata.standardConfig;
scs = double(cfg.SubcarrierSpacingkHz);
nrb = double(cfg.NSizeGrid);
tbs = double(cfg.TransportBlockSize);
rate = double(cfg.TargetCodeRate);
numSlots = 1;
if isfield(cfg, "NumSlots"), numSlots = double(cfg.NumSlots); end
rxNative = resampleToNative(rx, metadata);

pdsch = nrPDSCHConfig("Modulation",char(cfg.PDSCHModulation), "NumLayers",1, ...
    "PRBSet",0:(nrb-1), "SymbolAllocation",double(cfg.PDSCHSymbolAllocation));
pdsch.RNTI = 1;

% timing recovery from slot 0 DM-RS, then demodulate the whole (all-slot) grid
c0 = nrCarrierConfig("NCellID",1, "NSizeGrid",nrb, "SubcarrierSpacing",scs, "NSlot",0);
offset = nrTimingEstimate(c0, rxNative, nrPDSCHDMRSIndices(c0,pdsch), nrPDSCHDMRS(c0,pdsch));
if offset > 0 && offset < numel(rxNative)
    rxNative = rxNative(1+offset:end);
end
grid = nrOFDMDemodulate(c0, rxNative);

decodedBits = uint8([]);
referenceBits = uint8([]);
crcFails = 0;
for s = 0:numSlots-1
    cols = s*14 + (1:14);
    if cols(end) > size(grid, 2), break; end
    carrier = nrCarrierConfig("NCellID",1, "NSizeGrid",nrb, "SubcarrierSpacing",scs, "NSlot",s);
    slotGrid = grid(:, cols, :);
    dmrsInd = nrPDSCHDMRSIndices(carrier, pdsch);
    dmrsSym = nrPDSCHDMRS(carrier, pdsch);
    [hest, nVar] = nrChannelEstimate(carrier, slotGrid, dmrsInd, dmrsSym);
    pdschInd = nrPDSCHIndices(carrier, pdsch);
    [pr, ph] = nrExtractResources(pdschInd, slotGrid, hest);
    eq = nrEqualizeMMSE(pr, ph, nVar);
    cw = nrPDSCHDecode(carrier, pdsch, eq, nVar);
    dlsch = nrDLSCHDecoder;
    dlsch.TransportBlockLength = tbs;
    dlsch.TargetCodeRate = rate;
    [sb, crc] = dlsch(cw, pdsch.Modulation, 1, double(cfg.RV));
    crcFails = crcFails + (crc ~= 0);
    sb = uint8(sb(:));
    refStart = s*tbs;
    refSeg = txBits(refStart+1 : min(refStart+tbs, numel(txBits)));
    m = min(numel(sb), numel(refSeg));
    decodedBits = [decodedBits; sb(1:m)];        %#ok<AGROW>
    referenceBits = [referenceBits; refSeg(1:m)]; %#ok<AGROW>
end
if crcFails > 0
    warning("decode_waveforms_24576:FiveGCRC", "5G DL-SCH CRC failed for %d/%d slots in %s.", ...
        crcFails, numSlots, string(metadata.waveformName));
end
end

% ======================================================================= %
%                         802.11ax HE-SU RECEIVER                         %
% ======================================================================= %
function [decodedBits, referenceBits] = decodeWLAN(rx, metadata, txBits)
c = metadata.standardConfig;
cfg = wlanHESUConfig;
cfg.ChannelBandwidth = char(c.ChannelBandwidth);
cfg.MCS = double(c.MCS);
cfg.ChannelCoding = char(c.ChannelCoding);
cfg.APEPLength = double(c.APEPLengthBytes);
cfg.GuardInterval = double(c.GuardIntervalUs);
cfg.HELTFType = double(c.HELTFType);
cfg.NumTransmitAntennas = 1;
cfg.NumSpaceTimeStreams = 1;

cbw = cfg.ChannelBandwidth;
fs = wlanSampleRate(cfg);
rx = resampleToNative(rx, metadata);
ind = wlanFieldIndices(cfg);

% ---- packet detection + coarse CFO + fine timing + fine CFO ----
pktOffset = wlanPacketDetect(rx, cbw, 0, 0.5);
if isempty(pktOffset), pktOffset = 0; end

lstf = rx(pktOffset+(ind.LSTF(1):ind.LSTF(2)), :);
coarseCFO = wlanCoarseCFOEstimate(lstf, cbw);
rx = pfoCorrect(rx, fs, -coarseCFO);

searchEnd = min(pktOffset+ind.LSIG(2), size(rx,1));
nonht = rx(pktOffset+ind.LSTF(1):searchEnd, :);
finePkt = wlanSymbolTimingEstimate(nonht, cbw);
pktOffset = pktOffset + finePkt;

lltf = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)), :);
fineCFO = wlanFineCFOEstimate(lltf, cbw);
rx = pfoCorrect(rx, fs, -fineCFO);

% ---- HE-LTF channel estimate ----
rxHELTF = rx(pktOffset+(ind.HELTF(1):ind.HELTF(2)), :);
heltfDemod = wlanHEDemodulate(rxHELTF, "HE-LTF", cfg);
[chanEst, pilotEst] = wlanHELTFChannelEstimate(heltfDemod, cfg);

% ---- HE-Data equalize + bit recover ----
rxData = rx(pktOffset+(ind.HEData(1):ind.HEData(2)), :);
demod = wlanHEDemodulate(rxData, "HE-Data", cfg);
ofdmInfo = wlanHEOFDMInfo("HE-Data", cbw, cfg.GuardInterval);
noiseEst = wlanHEDataNoiseEstimate(demod(ofdmInfo.PilotIndices,:,:), pilotEst, cfg);
[eqSym, csi] = wlanHEEqualize(demod(ofdmInfo.DataIndices,:,:), ...
    chanEst(ofdmInfo.DataIndices,:,:), noiseEst, cfg, "HE-Data");
decodedBits = wlanHEDataBitRecover(eqSym, noiseEst, csi, cfg, ...
    "LDPCDecodingMethod","norm-min-sum", "EarlyTermination",true);
decodedBits = uint8(decodedBits(:));
referenceBits = txBits(1:min(numel(txBits), numel(decodedBits)));
decodedBits = decodedBits(1:numel(referenceBits));
end

function y = pfoCorrect(x, fs, cfoHz)
n = (0:size(x,1)-1).';
y = x .* exp(1j*2*pi*cfoHz/fs*n);
end

% ======================================================================= %
%                       GENERIC SINGLE-CARRIER                            %
% ======================================================================= %
function [decodedBits, referenceBits] = decodeSingleCarrier(rx, metadata, txBits)
rxNative = resampleToNative(rx, metadata);
cfg = metadata.singleCarrier;
sps = double(cfg.SamplesPerSymbol);
refSymbols = modSC(txBits, double(cfg.ModulationOrder), double(cfg.PhaseOffsetRad), char(cfg.SymbolOrder));

if strcmpi(char(cfg.FilterShape), "Normal")
    % Full raised-cosine TX pulse is already Nyquist (ISI-free): recover the
    % symbols by phase-search decimation, no second matched filter.
    [rxAligned, refAligned, refStart] = scPhaseSearch(rxNative, refSymbols, sps);
else
    % Square-root RC TX -> RRC matched receive filter -> RC composite (Nyquist).
    rxFilter = comm.RaisedCosineReceiveFilter( ...
        "Shape", char(cfg.FilterShape), "RolloffFactor", double(cfg.RolloffFactor), ...
        "FilterSpanInSymbols", double(cfg.FilterSpanInSymbols), ...
        "InputSamplesPerSymbol", sps, "DecimationFactor", sps);
    rxSymbols = rxFilter(rxNative);
    release(rxFilter);
    [rxAligned, refAligned, refStart] = alignByCorrelation(rxSymbols, refSymbols);
end
if isfield(metadata,"equalizer") && strcmpi(char(metadata.equalizer),"mmse")
    rxEq = mmseEqualizeDA(rxAligned, refAligned, 21);   % data-aided linear MMSE (removes ISI)
else
    h = estimateComplexGain(rxAligned, refAligned);
    rxEq = rxAligned ./ h;
end

decodedBits = demodSC(rxEq, double(cfg.ModulationOrder), double(cfg.PhaseOffsetRad), char(cfg.SymbolOrder));
bps = log2(double(cfg.ModulationOrder));
refBitStart = (refStart-1)*bps + 1;
refBitEnd = min(numel(txBits), refBitStart + numel(decodedBits) - 1);
referenceBits = txBits(refBitStart:refBitEnd);
decodedBits = decodedBits(1:numel(referenceBits));
end

% ======================================================================= %
%                            GENERIC OFDM                                 %
% ======================================================================= %
function [decodedBits, referenceBits] = decodeOFDM(rx, metadata, txBits)
% Drives the OFDMEndToEndExample streaming receiver (helperOFDMRxFrontEnd +
% helperOFDMRx) frame-by-frame with timingAdvance feedback. Every transmitted
% frame carries the same PN9 transport block, so each CRC-passing frame is
% compared against txBits.
ensureOFDMHelpers();
c = metadata.ofdm;
userParam = struct( ...
    "BWIndex", double(c.BWIndex), "modOrder", double(c.ModOrder), ...
    "codeRateIndex", double(c.CodeRateIndex), "numFrames", double(c.NumFrames), ...
    "numSymPerFrame", double(c.NumSymPerFrame), "fc", double(c.Fc), ...
    "enableCFO", false, "enableCPE", true, "enableFading", false, ...
    "chanVisual", false, "enableScopes", false, "verbosity", 0);
[sysParam, ~] = helperOFDMSetParameters(userParam);

% Reset the receiver's persistent state before processing this waveform.
clear helperOFDMRx helperOFDMRxSearch helperOFDMRxFrontEnd helperOFDMFrequencyOffset
rxObj = helperOFDMRxInit(sysParam);

rxNative = resampleToNative(rx, metadata);
symLen = sysParam.FFTLen + sysParam.CPLen;
frameLen = symLen * sysParam.numSymPerFrame;

% Pad to whole frames plus flush frames (the receiver needs the next frame's
% sync+reference symbols to channel-estimate the current frame).
nWhole = max(sysParam.numFrames, ceil(numel(rxNative)/frameLen));
rxNative = [rxNative; zeros((nWhole+2)*frameLen - numel(rxNative), 1)];

sysParam.timingAdvance = frameLen;
block = logical(txBits(:));
numChunks = floor(numel(rxNative)/frameLen);

decodedBits = [];
referenceBits = [];
for fr = 1:numChunks
    sysParam.frameNum = fr;
    chunk = rxNative((fr-1)*frameLen + (1:frameLen));
    rxIn = helperOFDMRxFrontEnd(chunk, sysParam, rxObj);
    try
        [bits, isConn, toff, dgn] = helperOFDMRx(rxIn, sysParam, rxObj);
    catch
        break;   % reached the zero-padded flush region (no next-frame reference)
    end
    sysParam.timingAdvance = toff;
    if isConn && ~isempty(bits)
        crcFail = false;
        if isfield(dgn,"dataCRCErrorFlag") && ~isempty(dgn.dataCRCErrorFlag)
            crcFail = logical(dgn.dataCRCErrorFlag(end));
        end
        if ~crcFail
            m = min(numel(bits), numel(block));
            decodedBits   = [decodedBits;   logical(bits(1:m))]; %#ok<AGROW>
            referenceBits = [referenceBits; block(1:m)];         %#ok<AGROW>
        end
    end
end
clear helperOFDMRx helperOFDMRxSearch helperOFDMRxFrontEnd helperOFDMFrequencyOffset

if isempty(decodedBits)
    % No CRC-passing frame: report a fully-errored decode rather than erroring.
    decodedBits = false(numel(block),1);
    referenceBits = block;
end
end

function ensureOFDMHelpers()
% Portable locator for the MathWorks "OFDM Transmitter and Receiver" example
% helpers (see generate_waveforms_24576.m for the search order).
if exist("helperOFDMRx","file") == 2, return; end
cands = string.empty;
e = string(getenv("OFDM_HELPER_DIR"));
if strlength(e) > 0, cands(end+1) = e; end
home = char(java.lang.System.getProperty("user.home"));
d = dir(fullfile(home, "Documents", "MATLAB", "Examples", "*", "comm", "OFDMEndToEndExample"));
for i = 1:numel(d), cands(end+1) = string(fullfile(d(i).folder, d(i).name)); end
cands(end+1) = string(fullfile(matlabroot, "examples", "comm", "OFDMEndToEndExample"));
for c = cands
    if isfolder(c), addpath(char(c)); end
    if exist("helperOFDMRx","file") == 2, return; end
end
error("decode_waveforms_24576:OFDMHelpers", ...
    ["helperOFDM* not found. Run openExample('comm/OFDMEndToEndExample') once " ...
     "to install the example, or set the OFDM_HELPER_DIR environment variable to " ...
     "its folder before calling this function."]);
end

% ======================================================================= %
%                            BLUETOOTH                                    %
% ======================================================================= %
function decodedBits = decodeBluetoothBREDR(rx, metadata)
c = metadata.standardConfig;
sps = double(c.SamplesPerSymbol);
rxNative = trimToMultiple(resampleToNative(rx, metadata), sps);
rxCfg = bluetoothPhyConfig("Mode",char(c.Mode), ...
    "SamplesPerSymbol",sps, "WhitenStatus",char(c.WhitenStatus));
decodedBits = bluetoothIdealReceiver(rxNative(:), rxCfg);
decodedBits = uint8(decodedBits(:));
end

function y = trimToMultiple(x, m)
x = x(:);
n = floor(numel(x)/m)*m;
y = x(1:n);
end

function decodedBits = decodeBluetoothLE(rx, metadata)
c = metadata.standardConfig;
rxNative = trimToMultiple(resampleToNative(rx, metadata), double(c.SamplesPerSymbol));
decodedBits = bleIdealReceiver(rxNative(:), "Mode",char(c.Mode), ...
    "SamplesPerSymbol",double(c.SamplesPerSymbol), "WhitenStatus",char(c.WhitenStatus), ...
    "ModulationIndex",double(c.ModulationIndex), "PulseLength",double(c.PulseLength), ...
    "NoiseVariance",1e-6);
decodedBits = uint8(decodedBits(:));
end

% ======================================================================= %
%                        FM AUDIO RECEIVER                                %
% ======================================================================= %
function result = receiveAudioFM(rx, metadata, opts, sourcePath)
rxNative = resampleToNative(rx, metadata);
nativeFs = double(metadata.nativeSampleRateHz);
freqDevHz = double(metadata.frequencyDeviationHz);
audioLPF = double(metadata.audioFm.AudioLowpassHz);

% FM demod by phase differentiation
msg = [0; angle(rxNative(2:end) .* conj(rxNative(1:end-1)))] * nativeFs/(2*pi*freqDevHz);
msg = real(msg(:)) - median(real(msg));
audioRec = lowpassAudio(msg, min(audioLPF, 0.45*nativeFs), nativeFs);
audioRec = audioRec - mean(audioRec);

% recover at native rate, then resample to compare rate
cmpFs = double(opts.AudioCompareRateHz);
[pa, qa] = rat(cmpFs/nativeFs, 1e-10);
audioOut = resample(audioRec, pa, qa);
audioOut = normAudio(audioOut);

% reference: regenerate the band-limited source segment used at TX
ref = referenceAudioSegment(metadata, nativeFs, numel(audioRec));
refOut = resample(ref, pa, qa);
refOut = normAudio(refOut);

[snrdB, corr] = audioQuality(refOut, audioOut);

if strlength(string(opts.OutputAudioFile)) > 0
    fp = string(opts.OutputAudioFile);
    d = fileparts(fp);
    if strlength(string(d)) > 0 && ~exist(d,"dir"), mkdir(d); end
    audiowrite(fp, max(min(audioOut,0.99),-0.99), cmpFs);
end

result = struct;
result.waveformName = string(metadata.waveformName);
result.class = string(metadata.class);
result.standard = string(metadata.standard);
result.sourcePath = string(sourcePath);
result.isFM = true;
result.numComparedBits = 0;
result.bitErrors = NaN;
result.BER = NaN;
result.audioSNRdB = snrdB;
result.audioCorr = corr;
result.audio = audioOut;
result.audioSampleRateHz = cmpFs;
result.channel = channelSummary(opts);
end

function ref = referenceAudioSegment(metadata, nativeFs, numSamples)
[audio, fs] = audioread(char(metadata.audioFm.AudioSourceFile));
if size(audio,2) > 1, audio = mean(audio,2); end
audio = double(audio(:)); audio(~isfinite(audio)) = 0;
pk = max(abs(audio)); if pk > 0, audio = audio./pk; end
lpf = double(metadata.audioFm.AudioLowpassHz);
if lpf < 0.45*fs, audio = lowpassAudio(audio, lpf, fs); end
startSec = 0;
if isfield(metadata.audioFm, "SourceStartSeconds")
    startSec = double(metadata.audioFm.SourceStartSeconds);
end
srcDur = numel(audio)/fs;
tTarget = (0:numSamples-1).'/nativeFs;
tSource = (0:numel(audio)-1).'/fs;
ref = interp1(tSource, audio, mod(startSec + tTarget, srcDur), "linear", 0);
ref = ref(:) - mean(ref);
end

function [snrdB, corr] = audioQuality(ref, rec)
ref = double(ref(:)); rec = double(rec(:));
n = min(numel(ref), numel(rec));
ref = ref(1:n); rec = rec(1:n);
% align (FM demod + filtering introduce a small delay) within +/- 2000 samples
maxLag = min(2000, n-1);
[c, lags] = xcorr(rec, ref, maxLag, "coeff");
[~, idx] = max(abs(c)); lag = lags(idx);
if lag > 0
    rec = rec(1+lag:end); ref = ref(1:numel(rec));
elseif lag < 0
    ref = ref(1-lag:end); rec = rec(1:numel(ref));
end
m = min(numel(ref), numel(rec)); ref = ref(1:m); rec = rec(1:m);
% scale recovered to best-fit reference, then signal-to-error ratio
a = (rec.'*ref)/(rec.'*rec + eps);
err = ref - a*rec;
snrdB = 10*log10(sum(ref.^2)/(sum(err.^2)+eps));
% normalized cross-correlation coefficient (manual, avoids corrcoef shapes)
corr = (ref.'*rec)/sqrt((ref.'*ref)*(rec.'*rec) + eps);
end

function y = normAudio(x)
x = x - mean(x); pk = max(abs(x));
if pk > 0, y = 0.95*x./pk; else, y = x; end
end

% ======================================================================= %
%                         CHANNEL + IO HELPERS                            %
% ======================================================================= %
function [rx, Fs, metadata, txBits, sourcePath] = loadDecodeInput(opts)
sourcePath = "";
if ischar(opts.target) || isstring(opts.target)
    sourcePath = string(opts.target);
    s = load(sourcePath, "f_sig", "Fs", "metadata", "txBits");
    rx = s.f_sig(:); Fs = s.Fs; metadata = s.metadata; txBits = s.txBits;
else
    rx = opts.target(:); Fs = opts.Fs; metadata = opts.Metadata; txBits = opts.TxBits;
    if isempty(Fs) || isempty(metadata)
        error("Vector input requires Fs and Metadata name-value args.");
    end
end
txBits = uint8(txBits(:));
end

function y = applyChannel(x, Fs, opts, metadata)
rng(opts.RandomSeed, "twister");
y = double(x(:));
n = (0:numel(y)-1).';

snr = opts.SNRdB; gain = opts.Gain; phase = opts.PhaseOffsetRad;
cfo = opts.FrequencyOffsetHz; mpD = opts.MultipathDelays; mpG = opts.MultipathGainsDb;

% "wireless" preset chooses class-appropriate impairments
if opts.Channel == "wireless"
    [snr, gain, phase, cfo, mpD, mpG] = wirelessPreset(metadata);
end

% multipath (simple tapped delay line)
if ~isempty(mpD)
    gains = 10.^(mpG(:)/20);
    h = zeros(max(mpD)+1, 1);
    h(mpD(:)+1) = gains;
    y = filter(h, 1, y);
end
% carrier frequency offset
if cfo ~= 0
    y = y .* exp(1j*2*pi*cfo/Fs*n);
end
% gain + phase
y = gain * exp(1j*phase) .* y;
% AWGN
if isfinite(snr)
    sigPow = mean(abs(y).^2);
    noisePow = sigPow / 10^(snr/10);
    y = y + sqrt(noisePow/2)*(randn(size(y)) + 1j*randn(size(y)));
end
end

function [snr, gain, phase, cfo, mpD, mpG] = wirelessPreset(metadata)
% A "simple wireless channel": flat fading (gain/phase) + AWGN for every class;
% standards-based receivers additionally get carrier frequency offset and a mild
% multipath profile (they perform real sync + per-subcarrier equalization). The
% generic single-carrier receiver uses only a single-tap equalizer, so it sees
% flat fading + AWGN (no delay spread, no CFO) -- the appropriate simple channel.
standard = string(metadata.standard);
gain = 0.7; phase = 0.3;
mpD = []; mpG = [];
switch standard
    case "5G NR downlink"
        % PDSCH-only (no SSB/PSS): receiver recovers timing from DM-RS but not
        % carrier frequency, so no CFO. Multipath + flat fading + AWGN. The SNR
        % is specified across the full 245.76 MHz band; wide channels (up to
        % ~98 MHz occupied) keep most of that noise after downsampling and their
        % large multi-code-block transport blocks need margin, so 50 dB is used
        % to exercise the full PDSCH/DL-SCH chain through multipath for every BW.
        snr = 50; cfo = 0;    mpD = [0 7 19]; mpG = [0 -9 -15];
    case "IEEE 802.11ax HE-SU"
        snr = 32; cfo = 1e3;  mpD = [0 7 19]; mpG = [0 -9 -15];
    case {"Bluetooth BR/EDR","Bluetooth LE"}
        snr = 25; cfo = 0; phase = 0; gain = 1;        % ideal Rx is sync-strict
    case "Generic single-carrier"
        snr = 30; cfo = 0;                             % single-tap eq -> flat fading
    case "Generic OFDM"
        snr = 28; cfo = 0; mpD = [0 5]; mpG = [0 -12]; % per-subcarrier eq handles delay spread
    case {"MP3 audio-source narrowband FM","MP3 audio-source broadband FM"}
        snr = 40; cfo = 0; phase = 0; gain = 1;
    otherwise
        snr = 30; cfo = 0;
end
end

function s = channelSummary(opts)
s = struct("Channel",opts.Channel, "SNRdB",opts.SNRdB, "Gain",opts.Gain, ...
    "PhaseOffsetRad",opts.PhaseOffsetRad, "FrequencyOffsetHz",opts.FrequencyOffsetHz);
end

function y = resampleToNative(rx, metadata)
if isfield(metadata, "resampling")
    p = double(metadata.resampling.P);
    q = double(metadata.resampling.Q);
else
    p = double(metadata.outputSampleRateHz/metadata.nativeSampleRateHz); q = 1;
end
if p == q
    y = double(rx(:));
else
    y = resample(double(rx(:)), q, p, 20);   % invert generator's resample
end
end

% ======================================================================= %
%                       MOD/DEMOD + ALIGN HELPERS                         %
% ======================================================================= %
function syms = modSC(bits, M, phaseOffset, symbolOrder)
bits = double(bits(:));
switch M
    case {2,4}
        syms = pskmod(bits, M, phaseOffset, symbolOrder, "InputType","bit");
    otherwise
        syms = qammod(bits, M, symbolOrder, "InputType","bit", "UnitAveragePower",true);
end
syms = syms(:);
end

function bits = demodSC(syms, M, phaseOffset, symbolOrder)
switch M
    case {2,4}
        bits = pskdemod(syms(:), M, phaseOffset, symbolOrder, "OutputType","bit");
    otherwise
        bits = qamdemod(syms(:), M, symbolOrder, "OutputType","bit", "UnitAveragePower",true);
end
bits = uint8(bits(:));
end

function [rxAligned, refAligned, refStart] = scPhaseSearch(rxNative, refSymbols, sps)
% Search the sps sampling phases; pick the one whose decimated stream best
% correlates with the PN9 reference (symbol timing recovery for Nyquist pulses).
rxNative = double(rxNative(:));
bestScore = -Inf; rxAligned = []; refAligned = []; refStart = 1;
for off = 1:sps
    cand = rxNative(off:sps:end);
    if numel(cand) < 8, continue; end
    [ra, rf, rs] = alignByCorrelation(cand, refSymbols);
    sc = abs(sum(ra .* conj(rf))) / sqrt(sum(abs(ra).^2)*sum(abs(rf).^2) + eps);
    if sc > bestScore
        bestScore = sc; rxAligned = ra; refAligned = rf; refStart = rs;
    end
end
if isempty(rxAligned), error("Single-carrier phase search failed."); end
end

function [rxAligned, refAligned, refStart] = alignByCorrelation(rxSymbols, refSymbols)
rxSymbols = double(rxSymbols(:)); refSymbols = double(refSymbols(:));
[c, lags] = xcorr(rxSymbols, refSymbols);
[~, idx] = max(abs(c)); lag = lags(idx);
if lag >= 0
    rxStart = lag+1; refStart = 1;
else
    rxStart = 1; refStart = 1-lag;
end
n = min(numel(rxSymbols)-rxStart+1, numel(refSymbols)-refStart+1);
if n <= 0, error("Unable to align decoded symbols to PN9 reference."); end
rxAligned = rxSymbols(rxStart:rxStart+n-1);
refAligned = refSymbols(refStart:refStart+n-1);
end

function h = estimateComplexGain(rx, ref)
rx = double(rx(:)); ref = double(ref(:));
den = sum(abs(ref).^2);
if den <= eps, h = 1; else
    h = sum(rx .* conj(ref))/den;
    if abs(h) <= eps, h = 1; end
end
end

function rxEq = mmseEqualizeDA(rx, ref, Ntaps)
% Data-aided symbol-spaced linear MMSE equalizer: train an FIR that maps the
% received symbols onto the known reference symbols (removes linear ISI that the
% single-tap gain cannot). Regularized so it does not enhance noise. Trained on
% the full known sequence -> a genie/upper-bound linear receiver.
rx = double(rx(:)); ref = double(ref(:));
N = min(numel(rx), numel(ref)); rx = rx(1:N); ref = ref(1:N);
if N < 8, h = sum(rx.*conj(ref))/max(sum(abs(rx).^2),eps); rxEq = rx*conj(h); return; end
L = min(Ntaps, 2*floor((N-2)/2)+1); if L < 1, L = 1; end
d = floor(L/2);
rxp = [zeros(d,1); rx; zeros(d,1)];
X = zeros(N, L);
for k = 1:L, X(:,k) = rxp(k:k+N-1); end
lambda = 1e-2 * (rx'*rx)/N;                 % MMSE regularization ~ noise floor
w = (X'*X + lambda*eye(L)) \ (X'*ref);
rxEq = X*w;
end

function y = lowpassAudio(x, cutoffHz, fs)
cutoffHz = min(cutoffHz, 0.45*fs);
try
    y = lowpass(x, cutoffHz, fs, "Steepness",0.85);
catch
    [b,a] = butter(6, cutoffHz/(fs/2));
    y = filtfilt(b,a,x);
end
end

% ======================================================================= %
%                              BATCH MODE                                 %
% ======================================================================= %
function tf = isBatchTarget(target)
target = string(target);
tf = isfolder(target) || endsWith(lower(target), ".csv");
end

function results = batchDecode(target, opts)
if isfolder(target)
    root = target;
    manifestPath = fullfile(root, "waveform_manifest.csv");
else
    manifestPath = target;
    root = fileparts(target);
end
if ~isfile(manifestPath)
    error("Manifest not found: %s", manifestPath);
end
manifest = readtable(manifestPath, TextType="string", Delimiter=",");

rows = cell(height(manifest),1);
fprintf("Decoding %d waveforms (channel: %s)\n", height(manifest), opts.Channel);
for i = 1:height(manifest)
    matPath = fullfile(root, manifest.matFile(i));
    o = opts; o.target = matPath; o.RandomSeed = 1000 + i;
    try
        r = decodeOne(o);
        rows{i} = struct("waveformName",string(r.waveformName), "class",string(r.class), ...
            "standard",string(r.standard), "isFM",double(r.isFM), ...
            "numComparedBits",r.numComparedBits, "bitErrors",toNum(r.bitErrors), ...
            "BER",toNum(r.BER), "audioSNRdB",toNum(r.audioSNRdB), "audioCorr",toNum(r.audioCorr), ...
            "status","ok");
        if r.isFM
            fprintf("  %-16s audioSNR=%6.2f dB corr=%5.3f  %s\n", r.class, r.audioSNRdB, r.audioCorr, r.waveformName);
        else
            fprintf("  %-16s BER=%-10.4g errors=%d/%d  %s\n", r.class, r.BER, r.bitErrors, r.numComparedBits, r.waveformName);
        end
    catch err
        rows{i} = struct("waveformName",manifest.waveformName(i), "class",manifest.class(i), ...
            "standard",manifest.standard(i), "isFM",NaN, "numComparedBits",NaN, ...
            "bitErrors",NaN, "BER",NaN, "audioSNRdB",NaN, "audioCorr",NaN, ...
            "status",string(err.message));
        fprintf("  %-16s ERROR: %s  (%s)\n", manifest.class(i), err.message, manifest.waveformName(i));
    end
end
results = struct2table([rows{:}]);

resPath = fullfile(root, "decode_results_" + char(opts.Channel) + ".csv");
writetable(results, resPath);

digital = results(results.isFM == 0 & results.status == "ok", :);
fm = results(results.isFM == 1 & results.status == "ok", :);
fprintf("\n==== SUMMARY (channel: %s) ====\n", opts.Channel);
if ~isempty(digital)
    fprintf("Digital: %d decoded, max BER = %.4g, mean BER = %.4g, %d with BER=0\n", ...
        height(digital), max(digital.BER), mean(digital.BER), sum(digital.BER==0));
end
if ~isempty(fm)
    fprintf("FM     : %d decoded, min audioSNR = %.2f dB, mean audioSNR = %.2f dB, min corr = %.3f\n", ...
        height(fm), min(fm.audioSNRdB), mean(fm.audioSNRdB), min(fm.audioCorr));
end
nbad = sum(results.status ~= "ok");
if nbad > 0, fprintf("Errors : %d waveform(s) failed to decode (see status column)\n", nbad); end
fprintf("Results: %s\n", resPath);
end

function v = toNum(x)
if isempty(x), v = NaN; else, v = double(x); end
end
