function dev_measure_cfo()
%DEV_MEASURE_CFO Is the residual carrier offset GLOBAL (one hardware LO error)
% or per-signal? Grid-search the best extra frequency offset for a spread of
% signals; if they cluster at one value it's a global CFO we can correct once.
addpath(fileparts(mfilename('fullpath')));
CAP = "/home/bqn82/captures/attenuation_dB_0";
GW  = "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG = 245760000;
m = jsondecode(fileread(CAP + ".sigmf-meta"));
A = m.annotations; if ~iscell(A), A = num2cell(A); end
manifest = readtable(fullfile(GW,"waveform_manifest.csv"), TextType="string", Delimiter=",");
mfMap = containers.Map(manifest.waveformName, 1:height(manifest));
fid = fopen(CAP + ".sigmf-data","r"); c = onCleanup(@() fclose(fid));

% pick a spread of narrowband/CFO-sensitive signals across the band
wf = {};
for k=1:numel(A)
    a=A{k};
    if isfield(a,"wfgt_kind") && strcmp(a.wfgt_kind,"waveform")
        wf{end+1}=a; %#ok<AGROW>
    end
end
idx = round(linspace(1, numel(wf), 12));
grid = -6e3:500:6e3;
fprintf("%-10s %-9s %9s %9s %8s\n","class","ctrMHz","bestOff","BER@best","BER@0");
for ii = idx
    a = wf{ii};
    cls = string(a.wfgt_class); v = string(a.wfgt_variation);
    gtCenter = 0.5*(a.core_freq_lower_edge + a.core_freq_upper_edge);
    if ~isKey(mfMap,char(v)), continue; end
    row = manifest(mfMap(char(v)),:);
    S = load(fullfile(GW,row.matFile),"metadata","txBits");
    md=S.metadata; tx=uint8(S.txBits(:)); native=double(md.nativeSampleRateHz);
    fseek(fid, a.core_sample_start*2*4, "bof");
    raw = fread(fid, 2*a.core_sample_count, "float32=>double");
    chunk = complex(raw(1:2:end), raw(2:2:end)); nn=(0:numel(chunk)-1).';
    best=inf; bestOff=0; ber0=NaN;
    for off = grid
        iq = chunk .* exp(-1j*2*pi*(gtCenter+off)/ORIG*nn);
        [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
        b = tryber(rx,native,md,tx);
        if off==0, ber0=b; end
        if b>=0 && b<best, best=b; bestOff=off; end
    end
    fprintf("%-10s %9.2f %8d %9.3g %8.3g\n", cls, gtCenter/1e6, bestOff, best, ber0);
end
end
function b = tryber(rx,Fs,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
