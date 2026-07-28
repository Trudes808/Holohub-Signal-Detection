function dev_debug_qpsk()
%DEV_DEBUG_QPSK Test whether a small center-frequency error (from quantized GT
% freq edges) is what kills narrowband decode. Reproduce with a controlled
% up/down-conversion of the clean waveform.
addpath(fileparts(mfilename('fullpath')));
GW = "/home/bqn82/holoscan_generated_waveform/generated_waveforms_24576";
mp = fullfile(GW,"QPSK","QPSK_Rs240kHz_fosf4_occ288kHz_phase0_gray_rrc_a0p20_pn9.mat");
S = load(mp,"f_sig","Fs","metadata","txBits");
f = S.f_sig(:); Fs = double(S.Fs); md = S.metadata; tx = uint8(S.txBits(:));
native = double(md.nativeSampleRateHz);
fprintf("QPSK Rs240k: native=%.0f Hz, symbolRate=%.0f\n", native, double(md.symbolRateHz));

% 1) clean decode straight from f_sig
r = decode_waveforms_24576(f,"Fs",Fs,"Metadata",md,"TxBits",tx,"Channel","none");
fprintf("clean f_sig BER = %.4g\n", r.BER);

Fc = 54.712e6;                       % place at this absolute center
n = (0:numel(f)-1).';
up = f .* exp(1j*2*pi*Fc/Fs*n);      % up-convert into the wideband

    function ber = extract_decode(fcEst)
        m = up .* exp(-1j*2*pi*fcEst/Fs*(0:numel(up)-1).');   % down-convert by estimate
        [P,Q] = rat(native/Fs,1e-12);
        rx = resample(m, P, Q);
        md2 = md; md2.resampling.P = 1; md2.resampling.Q = 1;
        rr = decode_waveforms_24576(rx,"Fs",native,"Metadata",md2,"TxBits",tx,"Channel","none");
        ber = rr.BER;
    end

fprintf("extract @ EXACT center           BER = %.4g\n", extract_decode(Fc));
for err = [2e3 6e3 12e3 24e3]
    fprintf("extract @ center + %5.0f Hz err  BER = %.4g\n", err, extract_decode(Fc+err));
end

% CFO search to recover from a 12 kHz center error
errCenter = Fc + 12e3;
best = inf; bestOff = 0;
for off = -30e3:1e3:30e3
    b = extract_decode(errCenter + off);
    if b < best, best = b; bestOff = off; end
end
fprintf("with 12kHz error, CFO search best BER = %.4g at extra offset %.0f Hz\n", best, bestOff);
end
