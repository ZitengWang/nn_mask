function [path,logpost]=viterbi(loglik,loginitp,logfinalp,logtransp)

% VITERBI Viterbi algorithm
% 
% path=viterbi(loglik,loginitp,logfinalp,logtransp)
%
% Inputs:
% loglik: nstates x nfram matrix of log-likelihood values
% loginitp: nstates x 1 vector of initial log-probability values
% logfinalp: nstates x 1 vector of final log-probability values
% logtransp: nstates x nstates matrix of transition log-probabilities
%
% Output:
% path: 1 x nfram best state sequence
% logpost: log-posterior probability of the best state sequence
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nstates,nfram]=size(loglik);

% Forward pass
logalpha=loglik(:,1)+loginitp;
prev=zeros(nstates,nfram-1);
for t=2:nfram,
    logalphaprev=logalpha;
    for n=1:nstates,
        [logalpha(n),prev(n,t-1)]=max(logalphaprev+logtransp(:,n));
    end
    logalpha=logalpha+loglik(:,t);
end

% Backward pass
path=zeros(1,nfram);
[logpost,path(nfram)]=max(logalpha+logfinalp);
for t=nfram-1:-1:1,
    path(t)=prev(path(t+1),t);
end

return