function ber_sweep_combine()
%BER_SWEEP_COMBINE Aggregate the per-level result files into BER-vs-attenuation
% (SNR) tables + figures, across detectors. Reads (from results/):
%   ber_<det>_<stem>_overall.csv   (one row per level, from ber_sweep_one)
%   ber_<det>_<stem>_byclass.csv   (per-class, from ber_eval_run)
% and the committed atten_0 outputs so the sweep includes 0 dB.
% Produces:
%   ber_sweep_overall.csv   wide: atten_dB, snr_dB, BER_<det>...
%   ber_sweep_byclass.csv   tidy: detector, atten_dB, snr_dB, class, BER, ...
%   ber_sweep_overall.png   overall BER vs SNR, one curve per detector
%   ber_sweep_byclass.png   7-class grid, BER vs SNR, one curve per detector
HERE = fileparts(mfilename('fullpath')); addpath(HERE);
resDir = fullfile(HERE, "results");
dets = ["ground_truth","coherent_power","finetuned_dino_m2"];
detLabel = ["ground truth","coherent power","dino ft (M2)"];

% ---- wide overall table, computed straight from the per-signal CSVs ----
% (bit-weighted overall BER = sum(bitErrors)/sum(numBits) over non-NaN rows;
% reproduces ber_eval_run.overallBER and picks up atten_0 automatically.)
d = dir(fullfile(resDir, "ber_*_attenuation_dB_*.csv"));
orows = {};
for i = 1:numel(d)
    nm = string(d(i).name);
    if endsWith(nm,"_byclass.csv") || endsWith(nm,"_overall.csv"), continue; end
    tok = regexp(nm, "^ber_(.+)_attenuation_dB_(\d+)\.csv$", "tokens", "once");
    if isempty(tok) || ~ismember(tok(1), dets), continue; end
    det = tok(1); L = str2double(tok(2));
    T = readtable(fullfile(d(i).folder, nm));
    be = T.bitErrors; nb = T.numBits;
    ber = sum(be(~isnan(be))) / max(1, sum(nb(~isnan(nb))));
    orows{end+1} = struct("detector",det, "atten_dB",L, "overallBER",ber); %#ok<AGROW>
end
overall = table();
if ~isempty(orows)
    allo = struct2table([orows{:}]);
    lv = unique(allo.atten_dB);
    overall = table(lv, 54-lv, 'VariableNames', {'atten_dB','snr_dB'});
    for det = dets
        col = "BER_"+det; overall.(char(col)) = nan(height(overall),1);
        sub = allo(allo.detector==det, :);
        for r = 1:height(sub)
            overall.(char(col))(overall.atten_dB==sub.atten_dB(r)) = sub.overallBER(r);
        end
    end
    overall = sortrows(overall, "atten_dB");
    writetable(overall, fullfile(resDir,"ber_sweep_overall.csv"));
    fprintf("==== overall BER vs attenuation ====\n"); disp(overall);
end

% ---- tidy per-class table from every byclass CSV present ----
d = dir(fullfile(resDir, "ber_*_attenuation_dB_*_byclass.csv"));
tidy = {};
for i = 1:numel(d)
    tok = regexp(string(d(i).name), "^ber_(.+)_attenuation_dB_(\d+)_byclass\.csv$", "tokens", "once");
    if isempty(tok) || ~ismember(tok(1), dets), continue; end
    det = tok(1); L = str2double(tok(2));
    Tb = readtable(fullfile(d(i).folder, d(i).name), TextType="string");
    for r = 1:height(Tb)
        tidy{end+1} = struct("detector",det, "atten_dB",L, "snr_dB",54-L, ...
            "class",Tb.class(r), "BER",Tb.BER(r), "detectRate",Tb.detectRate(r), ...
            "nDecoded",Tb.nDecoded(r), "nMiss",Tb.nMiss(r), "nInsuff",Tb.nInsuff(r)); %#ok<AGROW>
    end
end
tidyT = table();
if ~isempty(tidy)
    tidyT = sortrows(struct2table([tidy{:}]), ["detector","class","atten_dB"]);
    writetable(tidyT, fullfile(resDir,"ber_sweep_byclass.csv"));
end

% ---- figure: overall BER vs SNR ----
try
    if ~isempty(overall)
        fig = figure("Visible","off","Position",[100 100 900 560]); hold on; grid on;
        mk = ["-o","-s","-^"];
        lo = inf;
        for j = 1:numel(dets)
            col = "BER_"+dets(j);
            if ~ismember(col, string(overall.Properties.VariableNames)), continue; end
            y = overall.(char(col)); m = ~isnan(y); y(y<=0) = 1e-5;
            lo = min(lo, min(y(m)));
            plot(overall.snr_dB(m), y(m), mk(j), "LineWidth",1.6, "MarkerSize",6);
        end
        % data-driven y limits: a fixed [1e-5 1] wastes most of the axis once the
        % sweep spans ~0.02..1 (it was sized for the atten_0-only BERs).
        if ~isfinite(lo) || lo <= 0, lo = 1e-3; end
        set(gca,"YScale","log"); ylim([10^floor(log10(lo)) 1.2]); set(gca,"XDir","reverse");
        xlabel("SNR (dB)  [\approx 54 - attenuation]"); ylabel("overall bit-weighted BER");
        legend(detLabel, "Location","southwest"); title("Overall BER vs SNR by detector");
        exportgraphics(fig, fullfile(resDir,"ber_sweep_overall.png"), "Resolution",140); close(fig);
        fprintf("figure: %s\n", fullfile(resDir,"ber_sweep_overall.png"));
    end
catch e, fprintf("overall figure skipped: %s\n", e.message); end

% ---- figure: per-class grid ----
try
    if ~isempty(tidyT)
        classes = ["BPSK","QPSK","16QAM","OFDM","5G_Downlink","802_11ax","Bluetooth"];
        fig = figure("Visible","off","Position",[60 60 1400 800]);
        tl = tiledlayout(2,4,"TileSpacing","compact","Padding","compact");
        mk = ["-o","-s","-^"];
        ylo = min(tidyT.BER(tidyT.BER>0));                 % shared, data-driven floor
        if isempty(ylo) || ~isfinite(ylo), ylo = 1e-4; end
        ylo = 10^floor(log10(ylo));
        for ci = 1:numel(classes)
            nexttile; hold on; grid on;
            for j = 1:numel(dets)
                sub = tidyT(tidyT.detector==dets(j) & tidyT.class==classes(ci), :);
                if isempty(sub), continue; end
                sub = sortrows(sub,"snr_dB"); y = sub.BER; y(y<=0) = ylo;
                plot(sub.snr_dB, y, mk(j), "LineWidth",1.3, "MarkerSize",5);
            end
            set(gca,"YScale","log"); ylim([ylo 1.2]); set(gca,"XDir","reverse");
            title(strrep(classes(ci),"_","\_")); xlabel("SNR (dB)"); ylabel("BER");
        end
        % legend in the empty 8th tile (7 classes in a 2x4 grid) -- a "south"
        % horizontal legend renders with overlapping labels at this figure width.
        nexttile; axis off;
        hold on; mk2 = ["-o","-s","-^"];
        for j = 1:numel(dets), plot(NaN, NaN, mk2(j), "LineWidth",1.6, "MarkerSize",6); end
        legend(detLabel, "Location","north", "Box","on", "FontSize",11);
        title(tl, "Per-class BER vs SNR by detector");
        exportgraphics(fig, fullfile(resDir,"ber_sweep_byclass.png"), "Resolution",130); close(fig);
        fprintf("figure: %s\n", fullfile(resDir,"ber_sweep_byclass.png"));
    end
catch e, fprintf("byclass figure skipped: %s\n", e.message); end
end
