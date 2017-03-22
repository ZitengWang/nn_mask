function Y=apply_ir(A,R,del)

% APPLY_IR Filters a single-channel signal by a time-varying STFT-domain
% impulse response
%
% Y=apply_ir(A,R,del)
%
% Inputs:
% A: ntap x nchan x nbin x nblock multichannel impulse response in each
% frequency bin and each time block, with delays from del to del+ntap-1
% R: nbin x nfram STFT of the input signal
% del: minimum delay (0 for a causal filter)
%
% Output:
% Y: nbin x nfram x nchan STFT of the output signal
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

[ntap,nchan,nbin,nblock]=size(A);
[~,nfram]=size(R);

% Count significant STFT bins (above the median level)
Rmed=median(abs(R(1:end-1,:)),2);
Rabo=sum(abs(R(1:end-1,:)) > repmat(Rmed,[1 nfram]),1);

% Delimitate the time blocks
nabo=sum(Rabo)/nblock;
bpos=interp1(cumsum(Rabo)-.5*Rabo+1e-6*(1:nfram),1:nfram,(0:.5:nblock)*nabo);
bpos(1)=0;
bpos(end)=nfram;

% Loop over the blocks
Y=zeros(nbin,nfram,nchan);
for t=0:nblock-1,
    
    % Define the time window
    if nblock==1,
        bbeg=1;
        bend=nfram;
        win=ones(1,nfram);
    elseif t==0,
        bbeg=1;
        bmid=floor(bpos(2));
        bend=floor(bpos(4));
        win=[ones(1,bmid-bbeg+1) sin(.5*(bend-bmid+.5:2*(bend-bmid)-.5)/(bend-bmid)*pi).^2];
    elseif t==nblock-1,
        bbeg=floor(bpos(t*2))+1;
        bmid=floor(bpos(t*2+2));
        bend=nfram;
        win=[sin(.5*(.5:bmid-bbeg+1-.5)/(bmid-bbeg+1)*pi).^2 ones(1,bend-bmid)];
    else
        bbeg=floor(bpos(t*2))+1;
        bmid=floor(bpos(t*2+2));
        bend=floor(bpos(t*2+4));
        win=[sin(.5*(.5:bmid-bbeg+1-.5)/(bmid-bbeg+1)*pi).^2 sin(.5*(bend-bmid+.5:2*(bend-bmid)-.5)/(bend-bmid)*pi).^2];
    end
    Rblock=R(:,bbeg:bend);
    
    % Loop over the frequency bins
    for f=1:nbin,
        for d=del:del+ntap-1,
            if d>=0,
                Rblockd=[zeros(1,d) Rblock(f,1:end-d)];
            else
                Rblockd=[Rblock(f,1-d:end) zeros(1,-d)];
            end
            
            % Filtering
            Y(f,bbeg:bend,:)=Y(f,bbeg:bend,:)+permute(A(d-del+1,:,f,t+1)'*(win.*Rblockd),[3 2 1]);
        end
    end
end

return