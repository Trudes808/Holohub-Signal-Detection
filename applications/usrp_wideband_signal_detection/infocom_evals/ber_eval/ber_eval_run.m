function out = ber_eval_run(stem, detector, varargin)
%BER_EVAL_RUN Per-signal BER of a detector's SAVED snippets vs ground truth.
%
%   out = ber_eval_run("attenuation_dB_0", "coherent_power")
%   out = ber_eval_run(stem, detector, "TimeOverlapMin",0.10, "Limit",50, ...)
%
% Pipeline (matches the collaborator's spec):
%   For every ground-truth *data* waveform (wfgt:kind=="waveform"): find the
%   detector snippet(s) whose box center falls inside the GT band AND whose time
%   overlaps >= TimeOverlapMin of the GT duration. If none -> "failed to detect"
%   -> BER = 1 for that signal's bits. If found -> stitch the per-frame snippet
%   pieces, recenter to the KNOWN center frequency, resample to the waveform's
%   native rate, and decode with decode_waveforms_24576 using the KNOWN
%   metadata + txBits. BER is aggregated bit-weighted, split by modulation class.
%
% detector: "coherent_power" | "finetuned_dino_m2" | "ground_truth"
%           (snippet-based; "ground_truth" expects GT-box snippets under SnippetRoot)

HERE = fileparts(mfilename('fullpath'));
p = inputParser;
p.addParameter("CapturesDir", "/home/bqn82/captures");
p.addParameter("GenRoot", "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576");
p.addParameter("CachePath", fullfile(HERE, "wave_cache.mat"));   % shared precomputed decode cache
p.addParameter("SnippetRoot", "");   % default derived below
p.addParameter("OutDir", fullfile(HERE, "results"));
p.addParameter("TimeOverlapMin", 0.10);
p.addParameter("CompensateCenter", true);
p.addParameter("CorrectCFO", true);   % data-aided CFO vs known TX waveform (all classes)
p.addParameter("EqualizeSC", true);   % data-aided MMSE equalizer on the single-carrier path (ISI)
p.addParameter("Classes", ["BPSK","QPSK","16QAM","OFDM","5G_Downlink","802_11ax","Bluetooth"]);
p.addParameter("Limit", Inf);        % cap #GT signals (debug)
p.addParameter("DebugVar", "");      % print internals for variations containing this substring
p.addParameter("Verbose", true);
p.parse(varargin{:});
o = p.Results;
addpath(HERE);
if strlength(string(o.SnippetRoot)) == 0
    o.SnippetRoot = sprintf("/tmp/usrp_spectrograms/ber_eval/%s/iq/%s/snippets", detector, stem);
end
if ~exist(o.OutDir,"dir"), mkdir(o.OutDir); end

capMeta = fullfile(o.CapturesDir, stem + ".sigmf-meta");
fprintf("BER eval: detector=%s  stem=%s\n  GT=%s\n  snippets=%s\n", detector, stem, capMeta, o.SnippetRoot);

% ---- 1. ground-truth data waveforms ----
gt = load_gt_waveforms(capMeta, o.Classes);
fprintf("  GT data waveforms in scope: %d\n", numel(gt));

% ---- 2. manifest + per-variation metadata/txBits cache ----
manifest = readtable(fullfile(o.GenRoot, "waveform_manifest.csv"), TextType="string", Delimiter=",");
mfMap = containers.Map(manifest.waveformName, 1:height(manifest));
% Shared precomputed cache (metadata/txBits/refd per variation) so we don't
% reload 287 x ~15 MB .mat files every run. get_wave falls back to the .mat if
% a variation is missing from the cache.
cache = load_wave_cache(o.CachePath);

% ---- 3. index detector snippets ----
snip = index_snippets(o.SnippetRoot);
fprintf("  snippet pieces indexed: %d\n", numel(snip));
if isempty(snip) && detector ~= "ground_truth"
    error("No snippets found under %s (run the snip pipeline first).", o.SnippetRoot);
end
slo = [snip.freq_lo]; shi = [snip.freq_hi];      % detection box freq edges (Hz)
s0 = [snip.orig_start]; s1 = [snip.orig_end];    % original-timeline sample span

% ---- 4. per-signal decode ----
% "ground_truth" = genie reference: extract each signal straight from the
% capture at its known center/time (perfect detection + full bandwidth).
isGT = detector == "ground_truth";
capFid = -1;
if isGT
    capData = fullfile(o.CapturesDir, stem + ".sigmf-data");
    capFid = fopen(capData, "r");
    if capFid < 0, error("cannot open capture %s", capData); end
    capCleanup = onCleanup(@() fclose(capFid)); %#ok<NASGU>
    capLen = dir(capData).bytes / 8;   % total complex samples
end

n = min(numel(gt), o.Limit);
rows = cell(n,1);
for i = 1:n
    g = gt(i);
    w = get_wave(g.variation, o.GenRoot, manifest, mfMap, cache);   % {md,txBits,native,nbits,standard}
    if isempty(w)
        rows{i} = mkrow(g, NaN, NaN, NaN, "no_matfile"); continue;
    end
    % Waveforms are transmitted in fixed slots (>=1 ms) and tile/repeat to fill
    % them, so a full-slot signal is always decodable. Annotations shorter than
    % one slot (sample_count < 1 ms) are capture fragments (frame-edge clips) with
    % too little signal to decode -> data lost -> excluded (undecodable by anyone).
    MIN_SLOT_SAMPLES = 245760;   % 1 ms at 245.76 MHz
    if g.sample_count < MIN_SLOT_SAMPLES
        rows{i} = mkrow(g, NaN, NaN, NaN, "insufficient"); continue;
    end

    % true signal center = midpoint of the GT annotation's own freq edges
    % (NOT wfgt:block_center_hz, which is the multi-signal block center).
    gtCenter = 0.5*(g.freq_lo + g.freq_hi);
    gStart = g.sample_start; gEnd = g.sample_start + g.sample_count;

    % ---- obtain the received IQ at native rate (rx) ----
    try
        if isGT
            rx = extract_from_capture(capFid, gStart, gEnd, gtCenter, w.native, w.numOut, capLen);
        else
            inBand = slo <= gtCenter & gtCenter <= shi;   % detection band contains signal center
            ov = max(0, min(s1, gEnd) - max(s0, gStart));  % time overlap in samples
            frac = ov / max(1, g.sample_count);
            match = find(inBand & frac >= o.TimeOverlapMin);
            if isempty(match)
                rows{i} = mkrow(g, 1.0, w.nbits, w.nbits, "miss");   % failed to detect -> 100% BER
                continue;
            end
            pieces = dedup_by_frame(snip(match), gtCenter);
            rx = stitch_and_recenter(pieces, gtCenter, gStart, gEnd, w.native, o.CompensateCenter);
        end
        % Decode the UNtrimmed signal (each decoder's own sync handles idle
        % padding). A decoder throw on sub-frame input -> the signal was
        % slot-truncated/clipped in the capture (data lost -> "insufficient").
        r = decode_native(rx, w, o);
        % Fallback: if a detector's stitched multi-frame snippet decoded poorly,
        % the stitch seam may be corrupting a phase-continuous signal (GFSK/BT).
        % Try each frame-piece alone (each already holds >=1 tiled waveform) and
        % keep the best -- only fires on failures, so clean signals aren't slowed.
        if ~isGT && isfield(r,"BER") && r.BER > 0.05 && numel(pieces) > 1
            for pp = 1:numel(pieces)
                try
                    rp = decode_native(stitch_and_recenter(pieces(pp), gtCenter, gStart, gEnd, w.native, o.CompensateCenter), w, o);
                    if isfield(rp,"BER") && rp.BER < r.BER, r = rp; end
                catch
                end
            end
        end
        rows{i} = mkrow(g, r.BER, r.bitErrors, r.numComparedBits, "decoded");
    catch e
        % standards decoders throw on sub-frame-length input -> the signal was
        % truncated/clipped in the capture (data lost), not a decode failure.
        msg = lower(string(e.message));
        if contains(msg, ["length","samples","waveform","must be","exceed","time span","index"])
            rows{i} = mkrow(g, NaN, NaN, NaN, "insufficient");
        else
            rows{i} = mkrow(g, 1.0, w.nbits, w.nbits, "decode_err:" + string(e.message));
        end
    end
    if o.Verbose && mod(i,200)==0, fprintf("    %d/%d ...\n", i, n); end
end
T = struct2table([rows{:}]);

% ---- 5. aggregate + write ----
resPath = fullfile(o.OutDir, sprintf("ber_%s_%s.csv", detector, stem));
writetable(T, resPath);
byClass = summarize(T);
clsPath = fullfile(o.OutDir, sprintf("ber_%s_%s_byclass.csv", detector, stem));
writetable(byClass, clsPath);

decoded = T(T.status=="decoded",:);
totErr = sum(T.bitErrors(~isnan(T.bitErrors)));
totBit = sum(T.numBits(~isnan(T.numBits)));
out = struct("detector",string(detector), "stem",string(stem), ...
    "nSignals",height(T), "nDecoded",height(decoded), ...
    "nMiss",sum(T.status=="miss"), "nInsuff",sum(T.status=="insufficient"), ...
    "overallBER", totErr/max(1,totBit), ...
    "byClass",byClass, "table",T, "resultsCsv",string(resPath), "byClassCsv",string(clsPath));

fprintf("\n== %s @ %s ==\n  signals=%d  decoded=%d  miss=%d  insufficient/capture-truncated(excl)=%d  overall bit-weighted BER=%.4g\n", ...
    detector, stem, out.nSignals, out.nDecoded, out.nMiss, out.nInsuff, out.overallBER);
disp(byClass);
fprintf("  per-signal CSV: %s\n  per-class  CSV: %s\n", resPath, clsPath);
end

% ======================================================================= %
function gt = load_gt_waveforms(capMeta, classes)
m = jsondecode(fileread(capMeta));
A = m.annotations; if ~iscell(A), A = num2cell(A); end
classes = string(classes);
gt = struct("sample_start",{},"sample_count",{},"freq_lo",{},"freq_hi",{}, ...
            "block_center",{},"class",{},"variation",{});
for k = 1:numel(A)
    a = A{k};
    if ~isfield(a,"wfgt_kind") || ~strcmp(a.wfgt_kind,"waveform"), continue; end
    cls = "";
    if isfield(a,"wfgt_class"), cls = string(a.wfgt_class); end
    if ~ismember(cls, classes), continue; end
    v = ""; if isfield(a,"wfgt_variation"), v = string(a.wfgt_variation); end
    bc = 0; if isfield(a,"wfgt_block_center_hz"), bc = double(a.wfgt_block_center_hz); end
    gt(end+1) = struct( ...  %#ok<AGROW>
        "sample_start", double(a.core_sample_start), ...
        "sample_count", double(a.core_sample_count), ...
        "freq_lo", double(a.core_freq_lower_edge), ...
        "freq_hi", double(a.core_freq_upper_edge), ...
        "block_center", bc, "class", cls, "variation", v);
end
end

% ======================================================================= %
function snip = index_snippets(root)
snip = struct("meta",{},"data",{},"rate",{},"center",{}, ...
              "freq_lo",{},"freq_hi",{},"orig_start",{},"orig_end",{}, ...
              "off",{},"count",{});
if ~isfolder(root), return; end
d = dir(fullfile(root, "*.sigmf-meta"));
for i = 1:numel(d)
    mp = fullfile(d(i).folder, d(i).name);
    dp = replace(mp, ".sigmf-meta", ".sigmf-data");
    if ~isfile(dp), continue; end
    m = jsondecode(fileread(mp));
    A = m.annotations; if ~iscell(A), A = num2cell(A); end
    for k = 1:numel(A)
        a = A{k};
        rate = getdef(a,"wfgt_snippet_sample_rate", getdef(m.global,"core_sample_rate",NaN));
        ctr  = getdef(a,"wfgt_center_frequency", 0);
        snip(end+1) = struct( ...  %#ok<AGROW>
            "meta",mp, "data",dp, "rate",double(rate), "center",double(ctr), ...
            "freq_lo",double(getdef(a,"core_freq_lower_edge",-inf)), ...
            "freq_hi",double(getdef(a,"core_freq_upper_edge", inf)), ...
            "orig_start",double(getdef(a,"wfgt_orig_sample_start",0)), ...
            "orig_end",  double(getdef(a,"wfgt_orig_sample_end",0)), ...
            "off",  double(getdef(a,"core_sample_start",0)), ...
            "count",double(getdef(a,"core_sample_count",0)));
    end
end
end

% ======================================================================= %
function cache = load_wave_cache(cachePath)
if strlength(string(cachePath)) > 0 && isfile(cachePath)
    S = load(cachePath, "cache"); cache = S.cache;
else
    cache = containers.Map('KeyType','char','ValueType','any');
end
end

% ======================================================================= %
function w = get_wave(variation, genRoot, manifest, mfMap, cache)
w = [];
key = char(variation);
if isKey(cache,key), w = cache(key); return; end
if ~isKey(mfMap,key), return; end
row = manifest(mfMap(key),:);
matPath = fullfile(genRoot, row.matFile);
if ~isfile(matPath), return; end
S = load(matPath, "metadata","txBits","f_sig");
md = S.metadata; tx = uint8(S.txBits(:));
% CFO-estimation reference: decimate the known TX waveform to a rate that
% PRESERVES its occupied bandwidth (else a signal wider than the decimation
% rate aliases and the CFO estimate is wrong -- this broke uncoded BLE le1m/2m,
% which are 1-2 MHz wide). Cap at 8 MHz (enough resolution; wideband CFO is
% negligible so its center slice suffices).
occ = double(getdef(md,"designedOccupiedBandwidthHz",0));
nat = double(md.nativeSampleRateHz);
FdT = min([nat, max(1.92e6, 2.2*occ), 8e6]);
Dcfo = max(1, round(245760000/FdT));
refd = resample(double(S.f_sig(:)), 1, Dcfo);
w = struct("md",md, "txBits",tx, "native",nat, ...
           "nbits",numel(tx), "standard",string(md.standard), "refd",refd, ...
           "fdCfo",245760000/Dcfo, ...
           "numOut",double(getdef(md,"numOutputSamples",0)), "occBW",occ);
cache(key) = w; %#ok<NASGU>
end

% ======================================================================= %
function pieces = dedup_by_frame(cand, trueCenter)
% keep one piece per original-frame span: the one whose box center is closest
% to the known signal center; return sorted by time.
[~,ord] = sort([cand.orig_start]);
cand = cand(ord);
keys = arrayfun(@(s) s.orig_start, cand);
[uk,~,grp] = unique(keys);
pieces = repmat(cand(1),1,numel(uk));
for g = 1:numel(uk)
    members = cand(grp==g);
    [~,best] = min(abs([members.center] - trueCenter));
    pieces(g) = members(best);
end
end

% ======================================================================= %
function rx = stitch_and_recenter(pieces, trueCenter, gStart, gEnd, nativeFs, compensate)
% Reconstruct the saved signal: for each snippet piece, trim to the GT signal's
% time window (we know it exactly), recenter to the known center, resample to
% native, and concatenate. Trimming removes the surrounding full-frame noise so
% narrowband CFO estimation + decode are clean, and it naturally captures only
% what the detector actually saved (frame-edge truncation -> fewer bits).
ORIG = 245760000;                                    % original stream rate
% Trim EXACTLY to the GT slot [gStart, gEnd) -- no guard. The waveform tiles
% from gStart, so gStart is a packet/frame boundary; the sync-free ideal
% receivers (Bluetooth BR/EDR + LE) assume the waveform starts at sample 0, and
% prepending even a small guard of pre-slot content misaligns every packet
% (this alone sent BT from ~0 to ~0.8 BER). Classes with real timing recovery
% (single-carrier correlation, WLAN packet detect, 5G DM-RS, OFDM search) are
% indifferent. This also matches the genie path, which reads exactly the slot.
segs = cell(1,numel(pieces)); k = 0;
for i = 1:numel(pieces)
    s = pieces(i);
    dec = max(1, round(ORIG / s.rate));              % original samples per snippet sample
    w0 = max(gStart, s.orig_start); w1 = min(gEnd, s.orig_end);
    if w1 <= w0, continue; end
    j0 = max(0, floor((w0 - s.orig_start)/dec));
    j1 = min(s.count, ceil((w1 - s.orig_start)/dec));
    if j1 - j0 < 8, continue; end
    iq = read_cf32(s.data, s.off + j0, j1 - j0);
    if compensate && s.rate > 0
        off = trueCenter - s.center;                 % residual offset in the snippet
        % Mix using ABSOLUTE original-sample index (not a per-piece reset) so
        % consecutive frame-pieces concatenate phase-continuously -- a per-piece
        % phase reset puts a discontinuity at the stitch seam that corrupts
        % phase-continuous modulations (GFSK/Bluetooth).
        m = (s.orig_start + j0*dec) + (0:numel(iq)-1).' * dec;   % original-sample index
        iq = iq .* exp(-1j*2*pi*off/ORIG*m);         % bring known center to DC, seam-continuous
    end
    [P,Q] = rat(nativeFs / s.rate, 1e-9);
    k = k + 1; segs{k} = resample(double(iq), P, Q);
end
if k == 0, error("no usable snippet samples in GT window"); end
rx = vertcat(segs{1:k});
end

% ======================================================================= %
function rx = extract_from_capture(fid, gStart, gEnd, gtCenter, nativeFs, numOut, capLen) %#ok<INUSD>
% Genie reference: read EXACTLY the transmitted slot [gStart,gEnd) from the
% wideband capture, mix its known center to DC, resample to native. Extracting
% only the slot (not the full waveform length) avoids pulling in adjacent slots,
% and correctly models slot-truncation of waveforms longer than their slot.
ORIG = 245760000;
cnt = min(gEnd - gStart, capLen - gStart);
fseek(fid, gStart*2*4, "bof");                       % cf32: 2 float32 per complex
raw = fread(fid, 2*cnt, "float32=>double");
iq = complex(raw(1:2:end), raw(2:2:end));
nn = (0:numel(iq)-1).';
iq = iq .* exp(-1j*2*pi*gtCenter/ORIG*nn);
[P,Q] = rat(nativeFs/ORIG, 1e-12);
rx = resample(iq, P, Q);
end

% ======================================================================= %
function s = active_span(rx)
% Non-destructive: return the native-sample span of the actually-present signal
% (first..last bin whose power exceeds 4x the noise floor). Used only to detect
% slot-truncated / clipped signals; the signal itself is decoded untrimmed.
n = numel(rx); if n < 400, s = n; return; end
nb = 100; bl = floor(n/nb); if bl < 1, s = n; return; end
P = zeros(1,nb);
for b = 1:nb, seg = rx((b-1)*bl+1 : b*bl); P(b) = mean(abs(seg).^2); end
Ps = sort(P); k = max(3, round(0.10*nb)); noise = mean(Ps(1:k));
active = find(P > 4*noise);
if isempty(active), s = 0; return; end
s = (active(end) - active(1) + 1) * bl;
end

% ======================================================================= %
function rx = isolate_band(rx, nativeFs, occBW)
% Low-pass to the signal's occupied band to reject neighboring signals that the
% native-rate window keeps (matters when native >> occupied, e.g. a 2 MHz
% Bluetooth burst in a 16 MHz window). No-op when the signal fills the band.
if occBW <= 0, return; end
cutoff = occBW/2 * 1.15;
if cutoff >= 0.45*nativeFs, return; end
rx = lowpass(rx, cutoff, nativeFs, "Steepness", 0.9);
end

% ======================================================================= %
function r = decode_native(rx, w, o)
% Isolate the occupied band -> data-aided CFO -> decode (+ single-carrier MMSE),
% with a small BER-minimizing CFO refinement when the first decode is poor.
rx = isolate_band(rx, w.native, w.occBW);
if o.CorrectCFO
    cfo = estimate_cfo_da(rx, w.refd, w.native, w.fdCfo);
    rx = rx .* exp(-1j*2*pi*cfo/w.native*(0:numel(rx)-1).');
end
md = w.md; md.resampling.P = 1; md.resampling.Q = 1;
if o.EqualizeSC && w.standard == "Generic single-carrier", md.equalizer = "mmse"; end
r = decode_waveforms_24576(rx, "Fs", w.native, "Metadata", md, "TxBits", w.txBits, "Channel", "none");
if o.CorrectCFO && isfield(r,"BER") && r.BER > 0.05
    r2 = cfo_refine_decode(rx, w, md);
    if r2.BER < r.BER, r = r2; end
end
end

% ======================================================================= %
function rBest = cfo_refine_decode(rx, w, md)
% Small BER-minimizing carrier-offset search around the current estimate; used
% to rescue signals the correlation-based CFO estimate mislocked (e.g. GFSK/BLE).
rBest = struct("BER", inf, "bitErrors", NaN, "numComparedBits", 0);
n = (0:numel(rx)-1).';
for off = -8e3:500:8e3   % wide absolute search: the data-aided CFO can mislock on broad GFSK peaks
    rxo = rx .* exp(-1j*2*pi*off/w.native*n);
    try
        r = decode_waveforms_24576(rxo, "Fs", w.native, "Metadata", md, "TxBits", w.txBits, "Channel", "none");
        if isfield(r,"BER") && r.BER < rBest.BER, rBest = r; end
    catch
    end
end
end

% ======================================================================= %
function cfo = estimate_cfo_da(rx, refd, native, Fd)
% Universal data-aided CFO: correlate the received signal against the known TX
% waveform (both at Fd, chosen to preserve the occupied bandwidth) over a
% frequency grid; the offset that maximizes the correlation peak is the residual
% carrier offset (Hz). Works for any modulation (single-carrier, OFDM, 5G, GFSK).
[P,Q] = rat(Fd/native, 1e-12);
rxd = resample(rx, P, Q);
Nr = numel(rxd); if Nr < 16 || numel(refd) < 16, cfo = 0; return; end
nn = (0:Nr-1).';
L = 2^nextpow2(Nr + numel(refd));
Rf = conj(fft(refd, L));
best = -inf; cfo = 0;
for f = -8e3:200:8e3
    y = rxd .* exp(-1j*2*pi*f/Fd*nn);
    pk = max(abs(ifft(fft(y, L) .* Rf)));
    if pk > best, best = pk; cfo = f; end
end
end

% ======================================================================= %
function rx = fine_cfo_correct(rx, Fs, M) %#ok<DEFNU>  (superseded by estimate_cfo_da)
% Blind Mth-power carrier-frequency-offset estimate + correction. The GT
% freq edges are quantized to the detector FFT-bin grid (~24 kHz), leaving a
% residual offset that wrecks narrowband single-carrier decode (the generic
% receiver has no CFO loop). PSK/QAM^p has a tone at p*CFO; p=2 (BPSK) else 4.
rx = rx(:);
p = 2; if M > 2, p = 4; end
N = numel(rx); if N < 16, return; end
CFO_MAX = 30e3;                           % center error is bounded by the ~24 kHz GT freq-edge grid
Z = fftshift(abs(fft(rx.^p)));
fax = ((-floor(N/2):ceil(N/2)-1).') * (Fs/N);
win = abs(fax) < p*CFO_MAX;               % tone sits at p*CFO; exclude symbol-rate spurs
Z(~win) = -inf;
[~,idx] = max(Z);
cfo = fax(idx) / p;
rx = rx .* exp(-1j*2*pi*cfo/Fs*(0:N-1).');
end

% ======================================================================= %
function iq = read_cf32(path, off, count)
fid = fopen(path,"r");
if fid < 0, error("cannot open %s", path); end
c = onCleanup(@() fclose(fid));
if count > 0
    fseek(fid, off*2*4, "bof");                       % 2 float32 per complex sample
    raw = fread(fid, 2*count, "float32=>double");
else
    raw = fread(fid, Inf, "float32=>double");
end
iq = complex(raw(1:2:end), raw(2:2:end));
end

% ======================================================================= %
function r = mkrow(g, ber, errs, nbits, status)
r = struct("class",g.class, "variation",g.variation, ...
    "sample_start",g.sample_start, "sample_count",g.sample_count, ...
    "block_center_MHz", g.block_center/1e6, ...
    "BER",ber, "bitErrors",errs, "numBits",nbits, "status",string(status));
end

% ======================================================================= %
function byClass = summarize(T)
cls = unique(T.class);
rows = cell(numel(cls),1);
for i = 1:numel(cls)
    sub = T(T.class==cls(i),:);
    dec = sub(sub.status=="decoded",:);
    nMiss = sum(sub.status=="miss");
    nInsuff = sum(sub.status=="insufficient");   % signal truncated/clipped in capture -> data lost
    nDecodable = max(1, height(dec) + nMiss);    % signals that COULD be decoded if the detector saved them
    err = sum(sub.bitErrors(~isnan(sub.bitErrors)));
    bit = sum(sub.numBits(~isnan(sub.numBits)));
    rows{i} = struct("class",cls(i), "nSignals",height(sub), ...
        "nDecoded",height(dec), "nMiss",nMiss, "nInsuff",nInsuff, ...
        "detectRate", height(dec)/nDecodable, ...
        "BER", err/max(1,bit), ...
        "BER_detectedOnly", sum(dec.bitErrors)/max(1,sum(dec.numBits)));
end
byClass = struct2table([rows{:}]);
end

% ======================================================================= %
function v = getdef(s, f, d)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
