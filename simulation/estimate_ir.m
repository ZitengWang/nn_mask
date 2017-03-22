function A=estimate_ir(R,X,blen,ntap,del)

% ESTIMATE_IR Estimates the time-varying STFT-domain impulse response
% between a single-channel close-field signal and a multichannel far-field
% signal by cutting them into nonuniform time blocks and performing
% least-squares estimation in each block
%
% A=estimate_impulse_response(R,X,blen,ntap,del)
%
% Inputs:
% R: nbin x nfram STFT of the single-channel close-field signal
% X: nbin x nfram x nchan STFT of the multichannel far-field signal
% blen: targeted average block length in samples
% ntap: filter length in frames
% del: minimum delay (0 for a causal filter)
%
% Output:
% A: ntap x nchan x nbin x nblock multichannel impulse response in each
% frequency bin and each time block, with delays from del to del+ntap-1
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

[nbin,nfram,nchan]=size(X);

% Count significant STFT bins (above the median level)
Rmed=median(abs(R(1:end-1,:)),2);
Rabo=sum(abs(R(1:end-1,:)) > repmat(Rmed,[1 nfram]),1);

% Delimitate the time blocks
nabo=blen/2;
nblock=round(sum(Rabo)/nabo);
nabo=sum(Rabo)/nblock;
bpos=interp1(cumsum(Rabo)-.5*Rabo+1e-6*(1:nfram),1:nfram,(0:.5:nblock)*nabo);
bpos(1)=0;
bpos(end)=nfram;

% Loop over the blocks
A=zeros(ntap,nchan,nbin,nblock);
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
    Xblock=X(:,bbeg:bend,:);
    
    % Loop over the frequency bins
    for f=1:nbin,
        G=zeros(ntap);
        D=zeros(nchan,ntap);
        for d1=del:del+ntap-1,
            if d1>=0,
                Rblock1=[zeros(1,d1) Rblock(f,1:end-d1)];
            else
                Rblock1=[Rblock(f,1-d1:end) zeros(1,-d1)];
            end
            
            % Compute the auto-correlation matrix
            for d2=del:del+ntap-1,
                if d2>=0,
                    Rblock2=[zeros(1,d2) Rblock(f,1:end-d2)];
                else
                    Rblock2=[Rblock(f,1-d2:end) zeros(1,-d2)];
                end
                G(d1-del+1,d2-del+1)=win.*Rblock1*Rblock2';
            end
            
            % Compute the cross-correlation vector
            for c=1:nchan,
                D(c,d1-del+1)=win.*Xblock(f,:,c)*Rblock1';
            end
        end
        
        % Derive the least-squares solution
        A(:,:,f,t+1)=G\D';
    end
end

return