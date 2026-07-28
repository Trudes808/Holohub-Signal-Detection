function ber_eval_all(stem, varargin)
%BER_EVAL_ALL Run the BER eval for ground_truth + coherent_power + finetuned_dino_m2
% on one capture and produce a per-class comparison CSV + figure.
%
%   ber_eval_all("attenuation_dB_0")
%   ber_eval_all(stem, "Detectors",["ground_truth","coherent_power","finetuned_dino_m2"])
HERE = fileparts(mfilename('fullpath'));
addpath(HERE);
p = inputParser;
p.addParameter("Detectors", ["ground_truth","coherent_power","finetuned_dino_m2"]);
p.addParameter("OutDir", fullfile(HERE,"results"));
p.addParameter("Limit", Inf);
p.parse(varargin{:});
o = p.Results;
if ~exist(o.OutDir,"dir"), mkdir(o.OutDir); end

% Build the shared per-waveform decode cache once (no-op if it already exists),
% so all detector runs reuse it instead of reloading the generated .mat files.
ber_precompute();

R = struct();
for det = o.Detectors
    fprintf("\n########## %s ##########\n", det);
    R.(matlab.lang.makeValidName(det)) = ber_eval_run(stem, det, "OutDir", o.OutDir, "Limit", o.Limit);
end

% ---- assemble a per-class comparison table ----
dets = o.Detectors;
classes = ["BPSK","QPSK","16QAM","OFDM","5G_Downlink","802_11ax","Bluetooth"];
rows = {};
for c = classes
    row = struct("class", c);
    any = false;
    for det = dets
        s = R.(matlab.lang.makeValidName(det));
        bc = s.byClass;
        idx = find(bc.class == c, 1);
        f = matlab.lang.makeValidName(det);
        if isempty(idx)
            row.("BER_"+f) = NaN; row.("det_"+f) = NaN;
        else
            row.("BER_"+f) = bc.BER(idx); row.("det_"+f) = bc.detectRate(idx); any = true;
        end
    end
    if any, rows{end+1} = row; end %#ok<AGROW>
end
comp = struct2table([rows{:}]);
compPath = fullfile(o.OutDir, "ber_comparison_" + stem + ".csv");
writetable(comp, compPath);
fprintf("\n==== per-class BER comparison (%s) ====\n", stem);
disp(comp);
fprintf("comparison CSV: %s\n", compPath);

% overall summary line
fprintf("\noverall bit-weighted BER:\n");
for det = dets
    s = R.(matlab.lang.makeValidName(det));
    fprintf("  %-20s BER=%.4g  detected=%d/%d  miss=%d\n", det, s.overallBER, s.nDecoded, s.nSignals, s.nMiss);
end

% ---- figure: grouped bars, BER per class per detector ----
try
    make_figure(comp, dets, stem, o.OutDir);
catch e
    fprintf("figure skipped: %s\n", e.message);
end
end

function make_figure(comp, dets, stem, outDir)
fig = figure("Visible","off","Position",[100 100 1100 520]);
classes = comp.class;
nc = numel(classes); nd = numel(dets);
Y = zeros(nc, nd);
for j = 1:nd
    Y(:,j) = comp.("BER_" + matlab.lang.makeValidName(dets(j)));
end
Y(Y<=0) = 1e-5;                     % floor for log display
b = bar(categorical(classes,classes), Y); %#ok<NASGU>
set(gca,"YScale","log"); ylim([1e-5 1]);
ylabel("Bit Error Rate (log)"); grid on;
legend(strrep(cellstr(dets),"_","\_"), "Location","northwest");
title(sprintf("Per-class BER by detector — %s", strrep(stem,"_","\_")));
png = fullfile(outDir, "ber_comparison_" + stem + ".png");
exportgraphics(fig, png, "Resolution",140); close(fig);
fprintf("figure: %s\n", png);
end
