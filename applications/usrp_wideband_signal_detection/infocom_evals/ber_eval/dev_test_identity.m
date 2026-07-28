function dev_test_identity()
%DEV_TEST_IDENTITY Confirm the efficient decode route: resample a decimated
% snippet straight to the waveform's NATIVE rate and decode there (metadata
% resampling forced to identity), vs the proven route (upsample to 245.76 MSps).
% Both should give BER 0 on a clean simulated frequency-snip.
addpath(fileparts(mfilename('fullpath')));
GW = "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
oversample = 0.10;
fprintf("%-8s %-10s %-10s\n","class","BER_24576","BER_native");
for c = ["BPSK","QPSK","16QAM","5G_Downlink","802_11ax"]
    d = dir(fullfile(GW,c,"*.mat")); if isempty(d), continue; end
    S = load(fullfile(d(1).folder,d(1).name),"f_sig","Fs","metadata","txBits");
    md = S.metadata; Fs = double(S.Fs); f = S.f_sig(:); tx = uint8(S.txBits(:));
    if c=="BPSK", fprintf("md fields: %s\n", strjoin(fieldnames(md),", ")); end
    nativeFs = double(md.nativeSampleRateHz);
    dec = max(1, round(Fs/nativeFs)); snipRate = Fs/dec;   % snippet ~ native rate
    lp = fir1(200, min(0.99, (0.45*snipRate)/(Fs/2)));
    snippet = conv(f, lp(:), "same"); snippet = snippet(1:dec:end);   % decimated snippet @ snipRate

    % route A: snippet -> 245.76 MSps, decode with real metadata resampling
    [P,Q]=rat(Fs/snipRate,1e-9); rxA = resample(snippet,P,Q);
    berA = ber(@() decode_waveforms_24576(rxA,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none"));

    % route B: snippet -> native, identity resampling in decoder
    [P,Q]=rat(nativeFs/snipRate,1e-9); rxB = resample(snippet,P,Q);
    mdB = md; mdB.resampling.P = 1; mdB.resampling.Q = 1;
    berB = ber(@() decode_waveforms_24576(rxB,"Fs",nativeFs,"Metadata",mdB,"TxBits",tx,"Channel","none"));

    fprintf("%-8s %-10s %-10s\n", c, num2str(berA), num2str(berB));
end
end
function b = ber(fn)
try, r=fn(); b=r.BER; catch e, b=-1; fprintf(2,"  err: %s\n",e.message); end
end
