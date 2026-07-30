function ber_sweep_one(stem, detector, outDir, snippetRoot, detectionTable)
%BER_SWEEP_ONE Run ber_eval_run for one (stem, detector) and write a per-level
% overall-summary row to results/ber_<detector>_<stem>_overall.csv.
%
% Writes ONLY its own per-level file (no shared CSV), so many of these can run
% concurrently across attenuation levels without racing. ber_eval_run also emits
% the per-signal + byclass CSVs as usual. ber_sweep_combine aggregates them.
%
% Thread count is capped (env BER_THREADS, default 6) so a pool of concurrent
% level-workers doesn't oversubscribe the cores.
t = str2double(getenv("BER_THREADS"));
if ~isnan(t) && t >= 1, maxNumCompThreads(round(t)); end

HERE = fileparts(mfilename('fullpath')); addpath(HERE);
stem = string(stem); detector = string(detector);
% Optional outDir/snippetRoot let a variant sweep (e.g. the 75 kHz + 1 ms snipper
% gate) write into its own results folder and read its own snippets, leaving the
% baseline results untouched.
if nargin < 3 || strlength(string(outDir)) == 0
    outDir = fullfile(HERE, "results");
end
extra = {};
if nargin >= 4 && strlength(string(snippetRoot)) > 0
    extra = {"SnippetRoot", string(snippetRoot)};
end
% results_v2: region-level mask-coverage detection table (replaces the legacy match rule)
if nargin >= 5 && strlength(string(detectionTable)) > 0
    extra = [extra, {"DetectionTable", string(detectionTable)}];
end
if ~exist(outDir, "dir"), mkdir(outDir); end
r = ber_eval_run(stem, detector, "OutDir", outDir, extra{:});
L = sscanf(char(stem), "attenuation_dB_%d");
row = struct("detector",detector, "atten_dB",L, "snr_dB",54-L, ...
    "overallBER",r.overallBER, "nSignals",r.nSignals, "nDecoded",r.nDecoded, ...
    "nMiss",r.nMiss, "nInsuff",r.nInsuff);
outCsv = fullfile(outDir, sprintf("ber_%s_%s_overall.csv", detector, stem));
writetable(struct2table(row), outCsv);
fprintf("[sweep] %s %s -> overall BER=%.4g  (row -> %s)\n", detector, stem, r.overallBER, outCsv);
end
