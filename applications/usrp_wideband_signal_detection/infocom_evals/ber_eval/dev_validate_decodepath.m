function dev_validate_decodepath()
%DEV_VALIDATE_DECODEPATH Prove the "decode a decimated snippet" path before we
% trust real operator snippets. For a few generated .mat waveforms we:
%   1) inspect the metadata contract (resampling, native rate, standard),
%   2) simulate a frequency-snip of the clean 245.76 MSps f_sig (mix->LPF->decimate)
%      to a realistic snippet rate ~ occupied_bw*(1+oversample),
%   3) resample that snippet back up to 245.76 MSps and decode with Channel="none",
% expecting BER 0 (clean -> any residual error is a bug in our snip/resample glue).
addpath(fileparts(mfilename('fullpath')));
GW = "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";

probes = [ ...
    "BPSK/BPSK_Rs240kHz_fosf4_occ288kHz_phase0_gray_rrc_a0p20_pn9.mat"; ...
    "QPSK/"; ...   % filled in below (first QPSK found)
    "16QAM/"; ...
    "5G_Downlink/"; ...
    "OFDM/" ];

% resolve the wildcard folders to a concrete first .mat
for i = 1:numel(probes)
    if endsWith(probes(i), "/")
        d = dir(fullfile(GW, probes(i) + "*.mat"));
        if isempty(d), probes(i) = ""; else, probes(i) = string(d(1).name); probes(i) = fullfile(char(extractBefore(probes(i)+"","_")), ""); end
    end
end

% simpler: just grab the first .mat in each class dir
classes = ["BPSK","QPSK","16QAM","OFDM","5G_Downlink","802_11ax","Bluetooth"];
oversample = 0.10;   % snipper adds ~oversample_percent beyond detected bandwidth

fprintf("%-12s %-26s %-18s %-10s %-10s %s\n","class","standard","native/snipRate","BERclean","BERsnip","note");
for c = classes
    d = dir(fullfile(GW, c, "*.mat"));
    if isempty(d), fprintf("%-12s (no .mat)\n", c); continue; end
    matPath = fullfile(d(1).folder, d(1).name);
    S = load(matPath, "f_sig","Fs","metadata","txBits");
    md = S.metadata; Fs = double(S.Fs); f = S.f_sig(:); tx = uint8(S.txBits(:));

    % --- metadata contract (print once per class) ---
    nativeFs = getfielddef(md, "nativeSampleRateHz", NaN);
    occBW    = getfielddef(md, "occupiedBandwidthHz", NaN);
    if isfield(md,"resampling"), rs = sprintf("P=%g Q=%g", md.resampling.P, md.resampling.Q); else, rs = "(none)"; end
    stdName  = getfielddef(md, "standard", "?");

    % --- 1) clean decode straight from f_sig (sanity: should be BER 0) ---
    berClean = tryBER(@() decode_waveforms_24576(f, "Fs",Fs, "Metadata",md, "TxBits",tx, "Channel","none"));

    % --- 2) simulate a frequency snip of the clean signal ---
    % occupied bandwidth -> snippet rate; integer decimation from 245.76 MSps.
    if ~isfinite(occBW) || occBW<=0, occBW = nativeFs; end
    targetRate = occBW*(1+oversample);
    dec = max(1, floor(Fs/targetRate));
    snipRate = Fs/dec;
    % windowed-sinc low-pass to snippet band, then decimate (mix is identity: f_sig is DC-centered)
    fc = 0.45*snipRate;                       % passband edge inside the decimated band
    lp = designLP(fc, Fs);
    fLP = conv(f, lp, "same");
    snippet = fLP(1:dec:end);                 % decimated saved snippet (rate = snipRate)

    % --- 3) resample snippet back to 245.76 MSps and decode ---
    [P,Q] = rat(Fs/snipRate, 1e-9);
    rx24576 = resample(snippet, P, Q);
    berSnip = tryBER(@() decode_waveforms_24576(rx24576, "Fs",Fs, "Metadata",md, "TxBits",tx, "Channel","none"));

    fprintf("%-12s %-26s %6.0f/%-9.0f %-10s %-10s dec=%d\n", ...
        c, stdName, nativeFs, snipRate, fmtber(berClean), fmtber(berSnip), dec);
end
end

function v = getfielddef(s, f, d)
if isfield(s, f), v = double(s.(f)); if ~isnumeric(s.(f)), v = string(s.(f)); end
    if isstring(s.(f)) || ischar(s.(f)), v = string(s.(f)); end
else, v = d; end
end

function lp = designLP(fc, fs)
% simple linear-phase FIR low-pass (Kaiser), odd length
n = 200;
lp = fir1(n, min(0.99, fc/(fs/2)));
lp = lp(:);
end

function b = tryBER(fn)
try
    r = fn();
    if isstruct(r) && isfield(r,"BER"), b = r.BER; else, b = NaN; end
catch e
    b = -1; fprintf(2, "   [decode error] %s\n", e.message);
end
end

function s = fmtber(b)
if b < 0, s = "ERR"; elseif isnan(b), s = "NaN"; else, s = sprintf("%.3g", b); end
end
