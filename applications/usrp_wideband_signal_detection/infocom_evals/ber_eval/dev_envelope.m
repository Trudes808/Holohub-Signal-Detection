function dev_envelope()
%DEV_ENVELOPE Is the failing signal truncated in the capture? Measure in-band
% power vs time around a failing 5G / BT-LE and compare the "signal-present"
% length to annCnt and numOutputSamples.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid)); flen=dir(CAP+".sigmf-data").bytes/8;
for spec=[struct("cls","5G_Downlink"), struct("cls","Bluetooth")]
    a=firstShort(A,mfMap,spec.cls); v=char(a.wfgt_variation); row=manifest(mfMap(v),:);
    S=load(fullfile(GW,row.matFile),"metadata"); md=S.metadata;
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples); occ=double(md.designedOccupiedBandwidthHz);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    % read [gs, gs+1.6*numOut] at ORIG, mix to DC, isolate band at ORIG (cheap: just look at power)
    n=min(round(1.6*numOut),flen-gs); fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*n,"float32=>double");
    iq=complex(raw(1:2:end),raw(2:2:end)); iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).');
    % low-pass to occupied band at ORIG rate to isolate the target signal
    iqf=lowpass(iq, occ*0.6, ORIG, "Steepness",0.85);
    % power in 20 bins across the window
    nb=20; bl=floor(numel(iqf)/nb); P=zeros(1,nb);
    for b=1:nb, seg=iqf((b-1)*bl+(1:bl)); P(b)=mean(abs(seg).^2); end
    P=P/max(P);
    annBin=round(a.core_sample_count/ (numel(iq)/nb) *  (ORIG/ORIG));  % annCnt in bins (samples at ORIG)
    numBin=round(numOut/(numel(iq)/nb));
    fprintf("\n%s  %s\n  annCnt=%d (bin~%d)  numOut=%d (bin~%d)  windowBins=%d\n", spec.cls, v(1:min(35,end)), ...
        a.core_sample_count, annBin, numOut, numBin, nb);
    fprintf("  norm power/bin: %s\n", num2str(round(P,2)));
    aboveHalf=find(P>0.5); fprintf("  signal present (P>0.5) in bins %d..%d of %d\n", min(aboveHalf), max(aboveHalf), nb);
end
end
function a=firstShort(A,mfMap,cls)
a=[];
for k=1:numel(A)
    x=A{k};
    if isfield(x,"wfgt_kind")&&strcmp(x.wfgt_kind,"waveform")&&isfield(x,"wfgt_class")&&strcmp(x.wfgt_class,char(cls))&&x.core_sample_count<60000&&isKey(mfMap,char(x.wfgt_variation)), a=x; return; end
end
end
