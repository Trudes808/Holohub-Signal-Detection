function dev_test_ble()
%DEV_TEST_BLE Hypothesis: BLE fails because the slot contains many tiled packets
% but txBits is one packet. Extract ONE waveform length (numOut) vs the whole
% slot and compare BER. If one-packet extraction fixes it, adopt min(slot,numOut).
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid)); flen=dir(CAP+".sigmf-data").bytes/8;
% pick BLE + a couple BR/EDR (control) at a >=1ms slot
picks=pickBT(A,mfMap);
fprintf("%-32s %8s %9s %10s %9s\n","BT variation","slotms","BER_slot","BER_1wave","numOutms");
for i=1:numel(picks)
    a=picks{i}; v=char(a.wfgt_variation); row=manifest(mfMap(v),:);
    S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig"); md=S.metadata; tx=uint8(S.txBits(:));
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples); occ=double(md.designedOccupiedBandwidthHz);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    ref=resample(double(S.f_sig(:)),1,round(ORIG/native));
    bSlot=decodeWin(fid,gs,min(a.core_sample_count,flen-gs),gtCenter,ORIG,native,occ,md,tx,ref);
    bOne =decodeWin(fid,gs,min(numOut,flen-gs),gtCenter,ORIG,native,occ,md,tx,ref);
    fprintf("%-32s %8.2f %9s %10s %9.3f\n", v(1:min(32,end)), a.core_sample_count/ORIG*1e3, fb(bSlot), fb(bOne), numOut/ORIG*1e3);
end
end
function picks=pickBT(A,mfMap)
picks={}; seen=containers.Map('KeyType','char','ValueType','logical');
for k=1:numel(A)
    a=A{k};
    if ~(isfield(a,"wfgt_kind")&&strcmp(a.wfgt_kind,"waveform")&&isfield(a,"wfgt_class")&&strcmp(a.wfgt_class,'Bluetooth')), continue; end
    if a.core_sample_count<245760, continue; end       % >=1ms slot
    v=char(a.wfgt_variation); base=regexprep(v,'_sps\d+.*','');
    if isKey(seen,base)||~isKey(mfMap,v), continue; end
    seen(base)=true; picks{end+1}=a; %#ok<AGROW>
    if numel(picks)>=6, return; end
end
end
function b=decodeWin(fid,gs,cnt,gtCenter,ORIG,native,occ,md,tx,ref)
try
    fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*cnt,"float32=>double"); iq=complex(raw(1:2:end),raw(2:2:end));
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).'); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    if occ>0 && occ/2*1.15<0.45*native, rx=lowpass(rx,occ/2*1.15,native,"Steepness",0.9); end
    Fd=1.92e6;[Pd,Qd]=rat(Fd/native,1e-12);rxd=resample(rx,Pd,Qd);refd=resample(ref,Pd,Qd);
    Nr=numel(rxd);L=2^nextpow2(Nr+numel(refd));Rf=conj(fft(refd,L));best=-inf;cfo=0;
    for f=-8e3:200:8e3,y=rxd.*exp(-1j*2*pi*f/Fd*(0:Nr-1).');p=max(abs(ifft(fft(y,L).*Rf)));if p>best,best=p;cfo=f;end,end
    rx=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
    md.resampling.P=1;md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none");b=r.BER;
catch e, b=-1; end
end
function s=fb(b), if b<0,s="ERR";else,s=sprintf("%.3g",b);end,end
