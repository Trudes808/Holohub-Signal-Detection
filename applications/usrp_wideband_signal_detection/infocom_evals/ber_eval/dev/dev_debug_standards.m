function dev_debug_standards()
%DEV_DEBUG_STANDARDS Is the standards-class genie-decode failure a residual CFO
% (from FFT-bin-quantized GT freq edges)? Genie-extract one signal per standard
% class from the capture and sweep an extra frequency offset; if some nonzero
% offset recovers BER, the center error is the culprit.
addpath(fileparts(mfilename('fullpath')));
CAP = "/home/bqn82/captures/attenuation_dB_0";
GW  = "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG = 245760000;

m = jsondecode(fileread(CAP + ".sigmf-meta"));
A = m.annotations; if ~iscell(A), A = num2cell(A); end
manifest = readtable(fullfile(GW,"waveform_manifest.csv"), TextType="string", Delimiter=",");
mfMap = containers.Map(manifest.waveformName, 1:height(manifest));
fid = fopen(CAP + ".sigmf-data","r"); c = onCleanup(@() fclose(fid));

for cls = ["5G_Downlink","OFDM","Bluetooth","802_11ax"]
    a = firstOf(A, cls);
    if isempty(a), fprintf("%s: none\n", cls); continue; end
    flo = a.core_freq_lower_edge; fhi = a.core_freq_upper_edge;
    gtCenter = 0.5*(flo+fhi);
    gStart = a.core_sample_start; cnt = a.core_sample_count;
    v = string(a.wfgt_variation);
    if ~isKey(mfMap,char(v)), fprintf("%s: no manifest\n",cls); continue; end
    row = manifest(mfMap(char(v)),:);
    S = load(fullfile(GW,row.matFile),"metadata","txBits");
    md = S.metadata; tx = uint8(S.txBits(:)); native = double(md.nativeSampleRateHz);

    fseek(fid, gStart*2*4, "bof");
    raw = fread(fid, 2*cnt, "float32=>double");
    chunk = complex(raw(1:2:end), raw(2:2:end));
    nn = (0:numel(chunk)-1).';

    fprintf("\n%s  %s  native=%.0f  gtCenter=%.4f MHz  BW=[%.2f,%.2f]MHz\n", ...
        cls, extractBetween(v,1,min(30,strlength(v))), native, gtCenter/1e6, flo/1e6, fhi/1e6);
    best = inf; bestOff = 0;
    for off = -20e3:2e3:20e3
        iq = chunk .* exp(-1j*2*pi*(gtCenter+off)/ORIG*nn);
        [P,Q] = rat(native/ORIG,1e-12);
        rx = resample(iq, P, Q);
        b = tryber(rx, native, md, tx);
        if off==0, fprintf("   off=    0 Hz  BER=%.4g\n", b); end
        if b < best, best = b; bestOff = off; end
    end
    fprintf("   >>> best BER=%.4g at offset %+d Hz\n", best, bestOff);
end
end

function a = firstOf(A, cls)
a = [];
for k=1:numel(A)
    x=A{k};
    if isfield(x,"wfgt_kind") && strcmp(x.wfgt_kind,"waveform") && isfield(x,"wfgt_class") && strcmp(x.wfgt_class,cls)
        a=x; return;
    end
end
end

function b = tryber(rx, Fs, md, tx)
try
    md.resampling.P=1; md.resampling.Q=1;
    r = decode_waveforms_24576(rx,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none");
    b = r.BER;
catch e
    b = -1;
end
end
