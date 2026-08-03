function dev_test_cfo_da()
%DEV_TEST_CFO_DA Validate a universal data-aided CFO estimate (correlate against
% the known transmit waveform f_sig) on the problem cases before wiring it in.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid));

% collect one CFO-sensitive signal of each interesting kind
picks = pick(A, ["OFDM","Bluetooth","5G_Downlink","16QAM","QPSK"]);
fprintf("%-12s %8s %10s %9s %9s %9s\n","class","ctrMHz","cfoEst","BER@0","BER@DA","status");
for i=1:numel(picks)
    a=picks{i}; cls=string(a.wfgt_class); v=string(a.wfgt_variation);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge);
    if ~isKey(mfMap,char(v)),continue;end
    row=manifest(mfMap(char(v)),:);
    S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig");
    md=S.metadata; tx=uint8(S.txBits(:)); native=double(md.nativeSampleRateHz);
    fseek(fid,a.core_sample_start*2*4,"bof"); raw=fread(fid,2*a.core_sample_count,"float32=>double");
    chunk=complex(raw(1:2:end),raw(2:2:end)); nn=(0:numel(chunk)-1).';
    iq=chunk.*exp(-1j*2*pi*gtCenter/ORIG*nn);
    [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    % reference at native from f_sig (invert generator resample 245.76->native)
    [Pr,Qr]=rat(native/ORIG,1e-12); ref=resample(double(S.f_sig(:)),Pr,Qr);
    cfo=estimate_cfo_da(rx,ref,native);
    ber0=tryber(rx,native,md,tx);
    rx2=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
    [berDA,st]=tryber2(rx2,native,md,tx);
    fprintf("%-12s %8.2f %10.1f %9.3g %9.3g  %s\n",cls,gtCenter/1e6,cfo,ber0,berDA,st);
end
end

function picks=pick(A,classes)
picks={}; got=containers.Map('KeyType','char','ValueType','logical');
for k=1:numel(A)
    a=A{k};
    if isfield(a,"wfgt_kind")&&strcmp(a.wfgt_kind,"waveform")&&isfield(a,"wfgt_class")
        cls=a.wfgt_class;
        if any(classes==string(cls)) && ~isKey(got,cls) && abs(a.wfgt_block_center_hz-a.core_freq_lower_edge*0)>=0
            % prefer a non -30 (block-center) placement to exercise CFO
            if abs(0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge)+30e6)>5e6
                picks{end+1}=a; got(cls)=true; %#ok<AGROW>
            end
        end
    end
end
end

function cfo=estimate_cfo_da(rx,ref,Fs)
% coarse CFO by grid search maximizing |xcorr(rx e^{-j2pi f n}, ref)| on a
% decimated (anti-aliased) version, then decode-free. Robust to timing + modulation.
Rest=min(Fs,2e6); d=max(1,floor(Fs/Rest));
rxd=rx(1:d:end); refd=ref(1:d:end); Fd=Fs/d;
n=(0:numel(rxd)-1).'; L=2^nextpow2(numel(rxd)+numel(refd));
Rf=conj(fft(refd,L));
best=-inf; bestf=0;
for f=-8e3:200:8e3
    y=rxd.*exp(-1j*2*pi*f/Fd*n);
    C=abs(ifft(fft(y,L).*Rf));
    pk=max(C);
    if pk>best, best=pk; bestf=f; end
end
cfo=bestf;
end

function b=tryber(rx,Fs,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function [b,st]=tryber2(rx,Fs,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER; st="ok";
catch e, b=-1; st="ERR:"+string(e.message); end
end
