function dev_diag_5g_bt()
%DEV_DIAG_5G_BT (1) 5G: is it a per-slot timing drift? decode slot-by-slot with
% per-slot timing re-estimation. (2) BT: why weak correlation? check occupied BW
% of the captured burst vs expected, and self-correlation sanity.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
manifest=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mfMap=containers.Map(manifest.waveformName,1:height(manifest));
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid));
flen=dir(CAP+".sigmf-data").bytes/8;

% ---- 5G: find a failing (short) 5G, inspect NumSlots + per-slot decode ----
a5=firstShort(A,mfMap,"5G_Downlink");
row=manifest(mfMap(char(a5.wfgt_variation)),:); S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig");
md=S.metadata; sc=md.standardConfig;
fprintf("5G: %s\n  annCnt=%d numOut=%d NumSlots=%s TBS=%s NSizeGrid=%s SCS=%s\n", string(a5.wfgt_variation), ...
    a5.core_sample_count, double(md.numOutputSamples), num2str(getf(sc,'NumSlots')), num2str(getf(sc,'TransportBlockSize')), num2str(getf(sc,'NSizeGrid')), num2str(getf(sc,'SubcarrierSpacingkHz')));

% ---- BT: extraction sanity ----
for cls=["Bluetooth"]
    ab=firstShort(A,mfMap,cls); v=char(ab.wfgt_variation);
    row=manifest(mfMap(v),:); S=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig");
    md=S.metadata; native=double(md.nativeSampleRateHz);
    gtCenter=0.5*(ab.core_freq_lower_edge+ab.core_freq_upper_edge);
    fprintf("\nBT: %s  native=%.0f  occ_bw=%.0f Hz  ann_band=[%.3f,%.3f]MHz center=%.3f\n", v, native, ...
        getf(md,'designedOccupiedBandwidthHz'), ab.core_freq_lower_edge/1e6, ab.core_freq_upper_edge/1e6, gtCenter/1e6);
    % extract full window at gtCenter, native
    g=round(0.5*double(md.numOutputSamples)); w0=max(0,ab.core_sample_start-g); n=min(flen-w0, 2*double(md.numOutputSamples));
    fseek(fid,w0*2*4,"bof"); raw=fread(fid,2*n,"float32=>double"); iq=complex(raw(1:2:end),raw(2:2:end));
    iq=iq.*exp(-1j*2*pi*gtCenter/ORIG*(0:numel(iq)-1).'); [P,Q]=rat(native/ORIG,1e-12); rx=resample(iq,P,Q);
    % measure captured occupied bandwidth (99% power)
    bw=occBW(rx,native);
    % self-corr sanity: clean f_sig at native vs itself, and captured vs f_sig
    ref=resample(double(S.f_sig(:)),1,round(ORIG/native));
    scpk=peakSNR(rx,ref); selfpk=peakSNR(ref,ref);
    fprintf("   captured 99%% BW=%.0f Hz   peakSNR(capture,ref)=%.1f   peakSNR(ref,ref)=%.1f\n", bw, scpk, selfpk);
end
end

function a=firstShort(A,mfMap,cls)
a=[];
for k=1:numel(A)
    x=A{k};
    if isfield(x,"wfgt_kind")&&strcmp(x.wfgt_kind,"waveform")&&isfield(x,"wfgt_class")&&strcmp(x.wfgt_class,char(cls))&&x.core_sample_count<60000&&isKey(mfMap,char(x.wfgt_variation))
        a=x; return; end
end
end
function v=getf(s,f), if isfield(s,f), v=double(s.(f)); else, v=NaN; end, end
function bw=occBW(x,Fs)
X=abs(fftshift(fft(x))).^2; X=X/sum(X); cum=cumsum(X); N=numel(X);
lo=find(cum>=0.005,1); hi=find(cum>=0.995,1); bw=(hi-lo)/N*Fs;
end
function s=peakSNR(x,ref)
x=x(:); ref=ref(:); L=2^nextpow2(numel(x)+numel(ref));
C=abs(ifft(fft(x,L).*conj(fft(ref,L)))); s=max(C)/median(C(C>0));
end
