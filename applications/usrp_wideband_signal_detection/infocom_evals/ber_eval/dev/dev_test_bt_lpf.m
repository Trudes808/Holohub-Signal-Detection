function dev_test_bt_lpf()
%DEV_TEST_BT_LPF The genie extract resamples to native but never low-passes to
% the signal's occupied BW, so narrowband signals (BT: 2 MHz in a 16 MHz native
% window) pick up neighbors. Add a low-pass to occupied-BW and re-decode.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid));
flen=dir(CAP+".sigmf-data").bytes/8;
picks=firstN(A,mfMap,"Bluetooth",5);
fprintf("%-30s %8s %9s %9s\n","BT variation","occBWMHz","BER_noLP","BER_LP");
for i=1:numel(picks)
    a=picks{i}; v=char(a.wfgt_variation); row=manifest(mfMap(v),:);
    S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig"); md=S.metadata; tx=uint8(S.txBits(:));
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples);
    occ=double(md.designedOccupiedBandwidthHz); gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge);
    ref=resample(double(S.f_sig(:)),1,round(ORIG/native));
    gs=a.core_sample_start; n=min(numOut,flen-gs);
    fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*n,"float32=>double"); iq=complex(raw(1:2:end),raw(2:2:end));
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).'); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    bNo=decodeCfo(rx,native,md,tx,ref);
    rxlp=lowpass(rx, min(0.49*native, occ*0.75), native, "Steepness",0.9);
    bLp=decodeCfo(rxlp,native,md,tx,ref);
    fprintf("%-30s %8.2f %9s %9s\n", v(1:min(30,end)), occ/1e6, fb(bNo), fb(bLp));
end
end
function picks=firstN(A,mfMap,cls,N)
picks={};
for k=1:numel(A)
    x=A{k};
    if isfield(x,"wfgt_kind")&&strcmp(x.wfgt_kind,"waveform")&&isfield(x,"wfgt_class")&&strcmp(x.wfgt_class,char(cls))&&isKey(mfMap,char(x.wfgt_variation))
        picks{end+1}=x; if numel(picks)>=N, return; end, end %#ok<AGROW>
end
end
function b=decodeCfo(rx,native,md,tx,ref)
try
    Fd=1.92e6; [Pd,Qd]=rat(Fd/native,1e-12); rxd=resample(rx,Pd,Qd); refd=resample(ref,Pd,Qd);
    Nr=numel(rxd); L=2^nextpow2(Nr+numel(refd)); Rf=conj(fft(refd,L)); best=-inf; cfo=0;
    for f=-8e3:200:8e3, y=rxd.*exp(-1j*2*pi*f/Fd*(0:Nr-1).'); pk=max(abs(ifft(fft(y,L).*Rf))); if pk>best,best=pk;cfo=f;end, end
    rx=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
    md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function s=fb(b), if b<0, s="ERR"; else, s=sprintf("%.3g",b); end, end
