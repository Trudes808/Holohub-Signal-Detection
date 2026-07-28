function dev_test_fulllen()
%DEV_TEST_FULLLEN The throwing 5G/WLAN/BT signals are annotated with a short
% window (49152 / 9830) but the real waveform is longer (numOutputSamples).
% Extract the FULL waveform length from gStart and confirm they now decode.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid));
flen=dir(CAP+".sigmf-data").bytes/8;   % total complex samples in the capture

want=["5G_Downlink","802_11ax","Bluetooth"]; got=containers.Map('KeyType','char','ValueType','double');
fprintf("%-12s %-8s %8s %9s %9s %9s\n","class","annCnt","numOut","BER_ann","BER_full","status");
for k=1:numel(A)
    a=A{k};
    if ~(isfield(a,"wfgt_kind")&&strcmp(a.wfgt_kind,"waveform")), continue; end
    cls=string(a.wfgt_class); if ~any(want==cls), continue; end
    if a.core_sample_count>60000, continue; end            % only the short (throwing) ones
    if isKey(got,char(cls))&&got(char(cls))>=2, continue; end
    v=string(a.wfgt_variation); if ~isKey(mfMap,char(v)), continue; end
    row=manifest(mfMap(char(v)),:);
    S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig"); md=S.metadata; tx=uint8(S.txBits(:));
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples);
    refd=resample(double(S.f_sig(:)),1,128);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    berAnn=decodeWin(fid,gs,a.core_sample_count,gtCenter,native,ORIG,md,tx,refd);
    n=min(numOut, flen-gs);
    berFull=decodeWin(fid,gs,n,gtCenter,native,ORIG,md,tx,refd);
    if isKey(got,char(cls)), got(char(cls))=got(char(cls))+1; else, got(char(cls))=1; end
    fprintf("%-12s %8d %9d %9s %9s\n",cls,a.core_sample_count,numOut,fmt(berAnn),fmt(berFull));
end
end
function b=decodeWin(fid,gs,cnt,gtCenter,native,ORIG,md,tx,refd)
try
    fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*cnt,"float32=>double");
    iq=complex(raw(1:2:end),raw(2:2:end)); nn=(0:numel(iq)-1).';
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*nn);
    [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    % data-aided CFO (same as harness)
    Fd=1.92e6; [Pd,Qd]=rat(Fd/native,1e-12); rxd=resample(rx,Pd,Qd);
    Nr=numel(rxd); if Nr>=16 && numel(refd)>=16
        L=2^nextpow2(Nr+numel(refd)); Rf=conj(fft(refd,L)); best=-inf; cfo=0;
        for f=-8e3:200:8e3
            y=rxd.*exp(-1j*2*pi*f/Fd*(0:Nr-1).');
            pk=max(abs(ifft(fft(y,L).*Rf))); if pk>best,best=pk;cfo=f;end
        end
        rx=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
    end
    md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function s=fmt(b), if b<0, s="ERR"; else, s=sprintf("%.3g",b); end, end
