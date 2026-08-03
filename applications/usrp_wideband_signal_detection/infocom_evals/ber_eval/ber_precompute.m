function ber_precompute(varargin)
%BER_PRECOMPUTE Build a shared per-waveform decode cache so every ber_eval_run
% (ground_truth / coherent_power / finetuned_dino_m2 / ...) avoids reloading the
% 287 x ~15 MB generated .mat files on every run.
%
% For each waveform variation it stores exactly the `w` struct get_wave returns:
%   md (full decode metadata), txBits, native, nbits, standard, refd (decimated
%   CFO reference), fdCfo, numOut, occBW.
% Crucially it does NOT store f_sig (the 15 MB field) — only the tiny decimated
% refd derived from it — so the cache is ~75 MB instead of ~4.3 GB, and it is
% loaded once per run (~1 s) instead of 287 file loads.
%
%   ber_precompute                       % -> ./wave_cache.mat
%   ber_precompute("Force",true)         % rebuild even if present
HERE = fileparts(mfilename('fullpath')); addpath(HERE);
p = inputParser;
% External waveform library (not in the repo); override with BER_GEN_ROOT per machine.
p.addParameter("GenRoot", env_or("BER_GEN_ROOT", ...
    "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576"));
p.addParameter("CachePath", fullfile(HERE, "wave_cache.mat"));
p.addParameter("Force", false);
p.parse(varargin{:}); o = p.Results;

if isfile(o.CachePath) && ~o.Force
    fprintf("cache exists (%s) — use Force=true to rebuild\n", o.CachePath); return;
end
ORIG = 245760000;
manifest = readtable(fullfile(o.GenRoot, "waveform_manifest.csv"), TextType="string", Delimiter=",");
cache = containers.Map('KeyType','char','ValueType','any');
t0 = tic; nmiss = 0;
for i = 1:height(manifest)
    v = char(manifest.waveformName(i));
    matPath = fullfile(o.GenRoot, manifest.matFile(i));
    if ~isfile(matPath), nmiss = nmiss + 1; continue; end
    S = load(matPath, "metadata","txBits","f_sig");
    md = S.metadata; tx = uint8(S.txBits(:));
    occ = double(getdef(md,"designedOccupiedBandwidthHz",0));
    nat = double(md.nativeSampleRateHz);
    FdT = min([nat, max(1.92e6, 2.2*occ), 8e6]);
    Dcfo = max(1, round(ORIG/FdT));
    refd = resample(double(S.f_sig(:)), 1, Dcfo);
    cache(v) = struct("md",md, "txBits",tx, "native",nat, "nbits",numel(tx), ...
        "standard",string(md.standard), "refd",refd, "fdCfo",ORIG/Dcfo, ...
        "numOut",double(getdef(md,"numOutputSamples",0)), "occBW",occ); %#ok<NASGU>
    if mod(i,50)==0, fprintf("  cached %d/%d (%.0fs)\n", i, height(manifest), toc(t0)); end
end
save(o.CachePath, "cache", "-v7.3");
d = dir(o.CachePath);
fprintf("wrote %d waveforms (%d missing .mat) -> %s (%.0f MB, %.0fs)\n", ...
    cache.Count, nmiss, o.CachePath, d.bytes/1e6, toc(t0));
end

function v = getdef(s, f, d)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function v = env_or(name, dflt)
% Environment override for a machine-specific data path, else the default.
v = string(getenv(name));
if strlength(v) == 0, v = string(dflt); end
end
