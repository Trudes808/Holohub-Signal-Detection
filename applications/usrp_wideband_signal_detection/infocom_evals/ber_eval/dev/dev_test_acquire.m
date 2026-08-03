function dev_test_acquire()
%DEV_TEST_ACQUIRE Data-aided joint CFO+timing acquisition against the known TX
% waveform (f_sig). Extract a generous window, correlate to find the true signal
% start + CFO, extract full length from there, decode. Should rescue 5G/BT that
% fail with the naive "extract numOut from gStart".
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid));
flen=dir(CAP+".sigmf-data").bytes/8;

% pick: 2 short 5G, 2 short BT (failing), + a long 5G control
picks=selectPicks(A,mfMap,manifest);
fprintf("%-11s %-30s %9s %9s %8s %9s\n","class","which","BER_naive","peakSNR","lagNat","BER_acq");
for i=1:numel(picks)
    P=picks{i}; a=P.a; md=P.md; tx=P.tx; native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    ref=resample(double(P.fsig),1,round(ORIG/native));   % f_sig 245.76M -> native
    % naive: numOut from gs
    bNaive=dec(readmix(fid,gs,min(numOut,flen-gs),gtCenter,ORIG),native,ORIG,md,tx,ref);
    % acquisition: wide window [gs-g, gs+numOut+g]
    g=round(0.5*numOut); w0=max(0,gs-g); w1=min(flen,gs+numOut+g);
    rxw=dmix(fid,w0,w1-w0,gtCenter,ORIG,native);         % wide window at native
    [lag,pk,snr,cfo]=acquire(rxw, ref, native);
    lagNat=lag;
    seg=rxw(max(1,lag):min(numel(rxw),lag+numel(ref)-1));
    seg=seg.*exp(-1j*2*pi*cfo/native*(0:numel(seg)-1).');
    bAcq=decodeSeg(seg,native,md,tx);
    fprintf("%-11s %-30s %9s %9.1f %8d %9s\n",string(a.wfgt_class),P.name(1:min(30,end)),f(bNaive),snr,lagNat-(gs-w0),f(bAcq));
end
end

function picks=selectPicks(A,mfMap,manifest)
GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
picks={}; want={"5G_Downlink",2,"Bluetooth",2}; cnt=containers.Map({'5G_Downlink','Bluetooth','5G_long'},{0,0,0});
for k=1:numel(A)
    a=A{k};
    if ~(isfield(a,"wfgt_kind")&&strcmp(a.wfgt_kind,"waveform")), continue; end
    cls=char(a.wfgt_class); v=char(a.wfgt_variation);
    if ~isKey(mfMap,v), continue; end
    short=a.core_sample_count<60000;
    tag='';
    if strcmp(cls,'5G_Downlink')&&short&&cnt('5G_Downlink')<2, tag='5G_Downlink';
    elseif strcmp(cls,'Bluetooth')&&short&&cnt('Bluetooth')<2, tag='Bluetooth';
    elseif strcmp(cls,'5G_Downlink')&&~short&&cnt('5G_long')<1, tag='5G_long'; end
    if isempty(tag), continue; end
    row=manifest(mfMap(v),:); S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig");
    picks{end+1}=struct("a",a,"md",S.metadata,"tx",uint8(S.txBits(:)),"fsig",double(S.f_sig(:)),"name",v); %#ok<AGROW>
    cnt(tag)=cnt(tag)+1;
    if cnt('5G_Downlink')>=2 && cnt('Bluetooth')>=2 && cnt('5G_long')>=1, break; end
end
end

function [lag,pk,snr,cfo]=acquire(rxw,ref,Fs)
Fd=1.92e6; dd=max(1,round(Fs/Fd)); rxd=rxw(1:dd:end); refd=ref(1:dd:end); Fdc=Fs/dd;
Nr=numel(rxd); L=2^nextpow2(Nr+numel(refd)); Rf=conj(fft(refd,L));
best=-inf; cfo=0; idxb=1;
for ff=-8e3:200:8e3
    y=rxd.*exp(-1j*2*pi*ff/Fdc*(0:Nr-1).');
    C=abs(ifft(fft(y,L).*Rf));
    [pkf,ix]=max(C);
    if pkf>best, best=pkf; cfo=ff; idxb=ix; end
end
% peak SNR relative to median correlation
y=rxd.*exp(-1j*2*pi*cfo/Fdc*(0:Nr-1).'); C=abs(ifft(fft(y,L).*Rf));
snr=best/median(C(C>0)); pk=best;
lag=round((idxb-1)*dd)+1;
end

function iq=readmix(fid,gs,cnt,gtCenter,ORIG)
fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*cnt,"float32=>double");
iq=complex(raw(1:2:end),raw(2:2:end)); iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).');
end
function rx=dmix(fid,gs,cnt,gtCenter,ORIG,native)
iq=readmix(fid,gs,cnt,gtCenter,ORIG); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
end
function b=dec(iq,native,ORIG,md,tx,ref) %#ok<INUSD>
[P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q); b=decodeSeg(rx,native,md,tx);
end
function b=decodeSeg(rx,native,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function s=f(b), if b<0, s="ERR"; else, s=sprintf("%.3g",b); end, end
