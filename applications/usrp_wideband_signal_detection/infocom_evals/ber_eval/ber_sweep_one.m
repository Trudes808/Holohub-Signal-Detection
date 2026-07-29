function ber_sweep_one(stem, detector)
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
r = ber_eval_run(stem, detector);
L = sscanf(char(stem), "attenuation_dB_%d");
row = struct("detector",detector, "atten_dB",L, "snr_dB",54-L, ...
    "overallBER",r.overallBER, "nSignals",r.nSignals, "nDecoded",r.nDecoded, ...
    "nMiss",r.nMiss, "nInsuff",r.nInsuff);
outCsv = fullfile(HERE, "results", sprintf("ber_%s_%s_overall.csv", detector, stem));
writetable(struct2table(row), outCsv);
fprintf("[sweep] %s %s -> overall BER=%.4g  (row -> %s)\n", detector, stem, r.overallBER, outCsv);
end
