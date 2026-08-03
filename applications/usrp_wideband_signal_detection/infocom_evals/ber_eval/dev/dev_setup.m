function dev_setup()
%DEV_SETUP Verify toolboxes and install the OFDM example helpers the decoder needs.
try
    v = ver('bluetooth');
    if ~isempty(v), fprintf('Bluetooth Toolbox: %s %s\n', v(1).Name, v(1).Version);
    else, disp('Bluetooth Toolbox: NOT found'); end
catch e
    fprintf('BT ver error: %s\n', e.message);
end

% quick functional BT check
try
    cfg = bluetoothPhyConfig('Mode','BR','SamplesPerSymbol',8);  %#ok<NASGU>
    disp('bluetoothPhyConfig: OK');
catch e
    fprintf('bluetoothPhyConfig FAILED: %s\n', e.message);
end

% OFDM example helpers
if exist('helperOFDMRx','file') == 2
    fprintf('helperOFDMRx already on path.\n');
else
    try
        openExample('comm/OFDMEndToEndExample');   % copies to ~/Documents/MATLAB/Examples/... (no output in -batch)
        disp('openExample(comm/OFDMEndToEndExample) ran.');
    catch e
        fprintf('openExample failed: %s | %s\n', e.identifier, e.message);
    end
    % locate + add the copied helper folder
    home = char(java.lang.System.getProperty('user.home'));
    d = dir(fullfile(home,'Documents','MATLAB','Examples','**','helperOFDMRx.m'));
    for i = 1:numel(d), addpath(d(i).folder); end
    if ~isempty(d), fprintf('helper folder: %s\n', d(1).folder); end
end
fprintf('helperOFDMRx present now? %d (2 == yes)\n', exist('helperOFDMRx','file'));
fprintf('helperOFDMRxFrontEnd present? %d\n', exist('helperOFDMRxFrontEnd','file'));
end
