function dev_debug_bt_snippet()
%DEV_DEBUG_BT_SNIPPET Why do BT detector snippets decode to ~0.8 while genie
% capture-extraction is ~0? Trace one BT signal per mode: decode the coherent
% snippet vs the same signal extracted from the capture, dumping snippet geometry.
addpath(fileparts(mfilename('fullpath')));
CAP="/home/bqn82/captures/attenuation_dB_0"; GW="/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
SNIP="/tmp/usrp_spectrograms/ber_eval/coherent_power/iq/attenuation_dB_0/snippets"; ORIG=245760000;
m=jsondecode(fileread(CAP+".sigmf-meta")); A=m.annotations; if ~iscell(A),A=num2cell(A);end
man=readtable(fullfile(GW,"waveform_manifest.csv"),TextType="string",Delimiter=","); mf=containers.Map(man.waveformName,1:height(man));
S=indexSnips(SNIP);
fid=fopen(CAP+".sigmf-data","r"); c=onCleanup(@()fclose(fid)); flen=dir(CAP+".sigmf-data").bytes/8;
want=["Bluetooth_br_dh1_sps8_pn9","Bluetooth_le500k_sps8_pn9","Bluetooth_le2m_sps8_pn9"];
for t=want
    a=findMatch(A,t); if isempty(a), fprintf("%s: no >=1ms ann\n",t); continue; end
    row=man(mf(char(t)),:); L=load(fullfile(GW,row.matFile),"metadata","txBits","f_sig");
    md=L.metadata; tx=uint8(L.txBits(:)); native=double(md.nativeSampleRateHz); occ=double(md.designedOccupiedBandwidthHz);
    gc=0.5*(a.core_freq_lower_edge+a.core_freq_upper_edge); gs=a.core_sample_start; ge=gs+a.core_sample_count;
    ref=resample(double(L.f_sig(:)),1,round(ORIG/native));
    % capture extraction (genie)
    n=min(a.core_sample_count,flen-gs); fseek(fid,gs*2*4,"bof"); raw=fread(fid,2*n,"float32=>double");
    iqc=complex(raw(1:2:end),raw(2:2:end)); iqc=iqc.*exp(-1j*2*pi*gc/ORIG*(0:numel(iqc)-1).'); [P,Q]=rat(native/ORIG,1e-12);
    bCap=decodeAt(resample(iqc,P,Q),native,occ,ref,md,tx);
    % matching coherent snippet
    idx=find([S.flo]<=gc & gc<=[S.fhi] & min([S.oe],ge)-max([S.os],gs) >= 0.10*a.core_sample_count);
    fprintf("\n%s  native=%.0f occ=%.0fHz slot=%.1fms  BER_capture=%s  #snip=%d\n", ...
        t, native, occ, a.core_sample_count/ORIG*1e3, fb(bCap), numel(idx));
    if isempty(idx), continue; end
    [~,bi]=min(abs([S(idx).center]-gc)); s=S(idx(bi));
    fp=fopen(s.data,"r"); fseek(fp,s.off*2*4,"bof"); raw=fread(fp,2*s.count,"float32=>double"); fclose(fp);
    iqfull=complex(raw(1:2:end),raw(2:2:end));
    % (a) whole snippet, (b) trimmed to the GT slot [gs,ge] (mirror the harness)
    bWhole=decodeAt(resamp(iqfull.*exp(-1j*2*pi*(gc-s.center)/s.rate*(0:numel(iqfull)-1).'),native,s.rate),native,occ,ref,md,tx);
    dec=max(1,round(ORIG/s.rate)); j0=max(0,floor((gs-s.os)/dec)); j1=min(s.count,ceil((ge-s.os)/dec));
    iqt=iqfull(j0+1:j1); iqt=iqt.*exp(-1j*2*pi*(gc-s.center)/s.rate*(0:numel(iqt)-1).');
    bTrim=decodeAt(resamp(iqt,native,s.rate),native,occ,ref,md,tx);
    fprintf("  snippet: boxBW=%.3fMHz(occ %.3fMHz) rate=%.0f dec=%d center=%.3fMHz nsamp=%d span=%.1fms\n    BER_whole=%s  BER_trim=%s (trimlen=%d)\n", ...
        (s.fhi-s.flo)/1e6, occ/1e6, s.rate, dec, s.center/1e6, s.count, (s.oe-s.os)/ORIG*1e3, fb(bWhole), fb(bTrim), j1-j0);
end
end

function b=decodeAt(rx,native,occ,ref,md,tx)
% isolate band + data-aided CFO + BER-refine, mirroring the harness
if occ>0 && occ/2*1.15<0.45*native, rx=lowpass(rx,occ/2*1.15,native,"Steepness",0.9); end
Fd=min([native,max(1.92e6,2.2*occ),8e6]);
[Pr,Qr]=rat(Fd/native,1e-12); refd=resample(ref,Pr,Qr);
[Pd,Qd]=rat(Fd/native,1e-12); rxd=resample(rx,Pd,Qd);
Nr=numel(rxd); L=2^nextpow2(Nr+numel(refd)); Rf=conj(fft(refd,L)); best=-inf; cfo=0;
for f=-8e3:200:8e3, y=rxd.*exp(-1j*2*pi*f/Fd*(0:Nr-1).'); p=max(abs(ifft(fft(y,L).*Rf))); if p>best,best=p;cfo=f;end,end
rx=rx.*exp(-1j*2*pi*cfo/native*(0:numel(rx)-1).');
b=dec1(rx,native,md,tx);
if b>0.05
    for off=-3e3:500:3e3, r2=dec1(rx.*exp(-1j*2*pi*off/native*(0:numel(rx)-1).'),native,md,tx); if r2>=0&&r2<b,b=r2;end,end
end
end
function b=dec1(rx,native,md,tx)
try, md.resampling.P=1; md.resampling.Q=1;
    r=decode_waveforms_24576(rx,"Fs",native,"Metadata",md,"TxBits",tx,"Channel","none"); b=r.BER;
catch, b=-1; end
end
function S=indexSnips(root)
S=struct("data",{},"rate",{},"center",{},"flo",{},"fhi",{},"os",{},"oe",{},"off",{},"count",{});
d=dir(fullfile(root,"*.sigmf-meta"));
for i=1:numel(d)
    mp=fullfile(d(i).folder,d(i).name); dp=replace(mp,".sigmf-meta",".sigmf-data"); if ~isfile(dp),continue;end
    mm=jsondecode(fileread(mp)); AA=mm.annotations; if ~iscell(AA),AA=num2cell(AA);end
    for k=1:numel(AA), a=AA{k};
        S(end+1)=struct("data",dp,"rate",double(gd(a,"wfgt_snippet_sample_rate",gd(mm.global,"core_sample_rate",NaN))), ...
            "center",double(gd(a,"wfgt_center_frequency",0)),"flo",double(gd(a,"core_freq_lower_edge",-inf)), ...
            "fhi",double(gd(a,"core_freq_upper_edge",inf)),"os",double(gd(a,"wfgt_orig_sample_start",0)), ...
            "oe",double(gd(a,"wfgt_orig_sample_end",0)),"off",double(gd(a,"core_sample_start",0)),"count",double(gd(a,"core_sample_count",0))); %#ok<AGROW>
    end
end
end
function a=findMatch(A,v)
a=[]; for k=1:numel(A), x=A{k}; if isfield(x,"wfgt_variation")&&strcmp(x.wfgt_variation,char(v))&&x.core_sample_count>=245760,a=x;return;end,end
end
function y=resamp(x,native,rate), [P,Q]=rat(native/rate,1e-9); y=resample(x,P,Q); end
function v=gd(s,f,d), if isfield(s,f)&&~isempty(s.(f)),v=s.(f);else,v=d;end,end
function s=fb(b), if b<0,s="ERR";else,s=sprintf("%.3g",b);end,end
