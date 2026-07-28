function dev_diag_5g()
%DEV_DIAG_5G For failing 5G: extract full length + data-aided CFO, then decode
% with NumSlots=2 (as configured) vs NumSlots=1 (slot 0 only) vs finer CFO.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid)); flen=dir(CAP+".sigmf-data").bytes/8;
picks=firstN(A,mfMap,"5G_Downlink",4,true);   % short/failing ones
fprintf("%-42s %8s %9s %9s %10s\n","5G variation","BER_ns2","BER_ns1","BER_fineCFO","cfoHz");
for i=1:numel(picks)
    a=picks{i}; v=char(a.wfgt_variation); row=manifest(mfMap(v),:);
    S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig"); md=S.metadata; tx=uint8(S.txBits(:));
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    n=min(numOut,flen-gs);
    fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*n,"float32=>double"); iq=complex(raw(1:2:end),raw(2:2:end));
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).'); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    ref=resample(double(S.f_sig(:)),1,round(ORIG/native));
    [cfo,~]=cfoDA(rx,ref,native,200); rxc=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
    b2=dec(rxc,native,md,tx);
    md1=md; md1.standardConfig.NumSlots=1; b1=dec(rxc,native,md1,tx);
    [cfoF,~]=cfoDA(rx,ref,native,25); rxf=rx.*exp(-1j*2*pi*cfoF/native*(0:numel(rx)-1).'); bF=dec(rxf,native,md,tx);
    fprintf("%-42s %8s %9s %11s %10.1f\n", v(1:min(42,end)), fb(b2), fb(b1), fb(bF), cfo);
end
end
function picks=firstN(A,mfMap,cls,N,short)
picks={};
for k=1:numel(A)
    x=A{k};
    if isfield(x,"wfgt_kind")&&strcmp(x.wfgt_kind,"waveform")&&isfield(x,"wfgt_class")&&strcmp(x.wfgt_class,char(cls))&&isKey(mfMap,char(x.wfgt_variation))
        if short && x.core_sample_count>=60000, continue; end
        picks{end+1}=x; if numel(picks)>=N, return; end, end %#ok<AGROW>
end
end
function [cfo,pk]=cfoDA(rx,ref,Fs,step)
Fd=1.92e6; [Pd,Qd]=rat(Fd/Fs,1e-12); rxd=resample(rx,Pd,Qd); refd=resample(ref,Pd,Qd);
Nr=numel(rxd); L=2^nextpow2(Nr+numel(refd)); Rf=conj(fft(refd,L)); best=-inf; cfo=0;
for f=-8e3:step:8e3, y=rxd.*exp(-1j*2*pi*f/Fd*(0:Nr-1).'); p=max(abs(ifft(fft(y,L).*Rf))); if p>best,best=p;cfo=f;end, end
pk=best;
end
function b=dec(rx,native,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function s=fb(b), if b<0, s="ERR"; else, s=sprintf("%.3g",b); end, end
