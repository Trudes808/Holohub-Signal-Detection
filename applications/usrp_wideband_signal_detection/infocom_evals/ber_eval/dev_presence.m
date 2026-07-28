function dev_presence()
%DEV_PRESENCE Measure actual signal presence for representative signals using the
% exact harness extraction path, to calibrate the truncation criterion.
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid)); flen=dir(CAP+".sigmf-data").bytes/8;
targets=["BPSK_Rs240kHz_fosf4_occ288kHz_phase0_gray_rrc_a0p20_pn9", ...
         "BPSK_Rs61p44MHz_fosf4_occ82p944MHz_phase0_gray_rrc_a0p35_pn9", ...
         "5G_Downlink_bw100MHz_scs30kHz_pdschQPSK_pn9"];
for t=targets
    a=findAnn(A,t); if isempty(a), fprintf("%s: not found\n",t); continue; end
    row=manifest(mfMap(char(t)),:); S=load(fullfile(GW,row.matFile),"metadata"); md=S.metadata;
    native=double(md.nativeSampleRateHz); numOut=double(md.numOutputSamples); occ=double(md.designedOccupiedBandwidthHz);
    gtCenter=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start;
    cnt=min(max(a.core_sample_count,numOut),flen-gs);
    fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*cnt,"float32=>double"); iq=complex(raw(1:2:end),raw(2:2:end));
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).'); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    if occ>0 && occ/2*1.15<0.45*native, rx=lowpass(rx,occ/2*1.15,native,"Steepness",0.9); end
    expNat=numOut*native/ORIG;
    nb=20; bl=floor(numel(rx)/nb); Pw=zeros(1,nb);
    for b=1:nb, Pw(b)=mean(abs(rx((b-1)*bl+1:b*bl)).^2); end
    Pw=Pw/max(Pw);
    fprintf("\n%s\n  annCnt=%d numOut=%d native=%.0f | numel(rx)=%d expNat=%.0f (rx/exp=%.2f)\n", ...
        t, a.core_sample_count, numOut, native, numel(rx), expNat, numel(rx)/expNat);
    fprintf("  norm power/bin(20): %s\n", num2str(round(Pw,2)));
end
end
function a=findAnn(A,v)
a=[];
for k=1:numel(A), x=A{k}; if isfield(x,"wfgt_variation")&&strcmp(x.wfgt_variation,char(v)), a=x; return; end, end
end
