function CHiME3_simulate_data(official)

% CHIME3_SIMULATE_DATA Creates simulated data for the 3rd CHiME Challenge
%
% CHiME3_simulate_data
% CHiME3_simulate_data(official)
%
% Input:
% official: boolean flag indicating whether to recreate the official
% Challenge data (default) or to create new (non-official) data
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('off','all')
warning

if nargin < 1,
    official=true;
end

addpath ../utils;
upath='/talc/multispeech/corpus/speech_recognition/CHiME3/data/audio/16kHz/isolated//'; % path to segmented utterances
upath_ext = '/talc/multispeech/calcul/users/zwang/Data/CHiME3/isolated_ext_robot//';
cpath='/talc/multispeech/corpus/speech_recognition/CHiME3/data/audio/16kHz/embedded//'; % path to continuous recordings
bpath='/talc/multispeech/corpus/speech_recognition/CHiME3/data/audio/16kHz/backgrounds//'; % path to noise backgrounds
apath='/talc/multispeech/corpus/speech_recognition/CHiME3/data/annotations//'; % path to JSON annotations
nchan=6;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen_sub=256; % STFT window length in samples
blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
wlen_add=1024; % STFT window length in samples for speaker localization
del=-3; % minimum delay (0 for a causal filter)

%% Create simulated training dataset from original WSJ0 data %%
if exist('equal_filter.mat','file'),
    load('equal_filter.mat');
else
    % Compute average power spectrum of booth data
    nfram=0;
    bth_spec=zeros(wlen_sub/2+1,1);
    sets={'tr05' 'dt05'};
    for set_ind=1:length(sets),
        set=sets{set_ind};
        mat=json2mat([apath set '_bth.json']);
        for utt_ind=1:length(mat),
            oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
            o=wavread([upath set '_bth/' oname '.CH0.wav']);
            O=stft_multi(o.',wlen_sub);
            nfram=nfram+size(O,2);
            bth_spec=bth_spec+sum(abs(O).^2,2);
        end
    end
    bth_spec=bth_spec/nfram;
    
    % Compute average power spectrum of original WSJ0 data
    nfram=0;
    org_spec=zeros(wlen_sub/2+1,1);
    olist=dir([upath 'tr05_org/*.wav']);
    for f=1:length(olist),
        oname=olist(f).name;
        o=wavread([upath 'tr05_org/' oname]);
        O=stft_multi(o.',wlen_sub);
        nfram=nfram+size(O,2);
        org_spec=org_spec+sum(abs(O).^2,2);
    end
    org_spec=org_spec/nfram;
    
    % Derive equalization filter
    equal_filter=sqrt(bth_spec./org_spec);
    save('equal_filter.mat','equal_filter');
end
if 0
% Read official annotations
if official,
    mat=json2mat([apath 'tr05_simu.json']);

% Create new (non-official) annotations
else
    envirs={'BUS' 'CAF' 'PED' 'STR'};
    envir=envirs{randperm(4,1)}; % draw a random environment
end

% Loop over utterances
for utt_ind=1:length(mat),
    if official,
        udir=[upath 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
        udir_ext=[upath_ext 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
    else
        udir=[upath 'tr05_' lower(mat{utt_ind}.environment) '_simu_new/'];
    end
    if ~exist(udir,'dir'),
        system(['mkdir -p ' udir]);
    end
    if ~exist(udir_ext,'dir'),
        system(['mkdir -p ' udir_ext]);
    end
    oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
    iname=mat{utt_ind}.ir_wavfile;
    nname=mat{utt_ind}.noise_wavfile;
    uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
    ibeg=round(mat{utt_ind}.ir_start*16000)+1;
    iend=round(mat{utt_ind}.ir_end*16000);
    nbeg=round(mat{utt_ind}.noise_start*16000)+1;
    nend=round(mat{utt_ind}.noise_end*16000);

    % Load WAV files
    o=wavread([upath 'tr05_org/' oname '.wav']);
    [r,fs]=wavread([cpath iname '.CH0.wav'],[ibeg iend]);
    x=zeros(iend-ibeg+1,nchan);
    n=zeros(nend-nbeg+1,nchan);
	ysimu=zeros(nend-nbeg+1,nchan);
    for c=1:nchan,
        x(:,c)=wavread([cpath iname '.CH' int2str(c) '.wav'],[ibeg iend]);
        n(:,c)=wavread([bpath nname '.CH' int2str(c) '.wav'],[nbeg nend]);
		ysimu(:,c)=o;
    end
    
    SNR = normrnd(5, 5)   %dB
    SNR = 10^(SNR/10);
    % Normalize level and add
    ysimu=sqrt(SNR/sum(sum(x.^2))*sum(sum(n.^2)))*ysimu;
    xsimu=ysimu+n;
    
    % Write WAV file
    for c=1:nchan,
   %     wavwrite(xsimu(:,c),fs,[udir uname '.CH' int2str(c) '.wav']);
        audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:, c),fs);
        audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu(:, c), fs);
    end
end
end

%% Create simulated development and test datasets from booth recordings %%
sets={'dt05'};
for set_ind=1:length(sets),
    set=sets{set_ind};

    % Read official annotations
    if official,
        mat=json2mat([apath set '_simu.json']);
        
    % Create new (non-official) annotations
    else
    end
    
    % Loop over utterances
    for utt_ind=1:length(mat),
        if official,
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu/'];
            udir_ext=[upath_ext 'dt05_' lower(mat{utt_ind}.environment) '_simu/'];
        else
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
        end
        if ~exist(udir,'dir'),
            system(['mkdir -p ' udir]);
        end
        if ~exist(udir_ext,'dir'),
            system(['mkdir -p ' udir_ext]);
        end
        oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
        nname=mat{utt_ind}.noise_wavfile;
        uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
        tbeg=round(mat{utt_ind}.noise_start*16000)+1;
        tend=round(mat{utt_ind}.noise_end*16000);
        
        % Load WAV files
        o=wavread([upath set '_bth/' oname '.CH0.wav']);
        [r,fs]=wavread([cpath nname '.CH0.wav'],[tbeg tend]);
        nsampl=length(r);
        x=zeros(nsampl,nchan);
		
        ysimu = zeros(nsampl, nchan);
        for c=1:nchan,
            x(:,c)=wavread([cpath nname '.CH' int2str(c) '.wav'],[tbeg tend]);
	    ysimu(:,c)=o;
        end
        
        SNR = normrnd(10, 5)   %dB
        SNR = 10^(SNR/10);
	noise_dir='/talc/multispeech/calcul/users/zwang/Data/robot_noise/';
	noise_files={'desk1' 'desk2' 'carpet1' 'carpet2' 'floor1' 'floor2'};
        noise_file=noise_files{randperm(6,1)}; % draw a random noise file
	noise=wavread([noise_dir noise_file '.wav']);
	noise=[noise(:,1) noise(:,3) noise(:,4)];
	noise_len=length(noise(:,1));
	n=zeros(nsampl,nchan);
	nbeg=round(rand*(noise_len-nsampl-16000));
	for c=1:3,
	    n(:,c)=noise(nbeg:nbeg+nsampl-1,c);
	    n(:,c+3)=1.2*noise(nbeg+8000:nbeg+nsampl+8000-1,c);
	end
			
        % Normalize level and add
        ysimu=sqrt(SNR/sum(sum(x.^2))*sum(sum(n.^2)))*ysimu;
        xsimu=ysimu+n;
        
        % Write WAV file
        for c=1:nchan,
        %    wavwrite(xsimu(:,c),fs,[udir uname '.CH' int2str(c) '.wav']);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:, c),fs);
            audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu(:, c), fs);
        end
    end
end

return
