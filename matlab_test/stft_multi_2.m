function X=stft_multi_2(x,wlen)

% STFT_MULTI Multichannel short-time Fourier transform (STFT) using
% half-overlapping sine windows.
%
% X=stft_multi(x)
% X=stft_multi(x,wlen)
%
% Inputs:
% x: nsampl x nchan matrix containing nchan time-domain mixture signals
% wlen: window length (default: 1024 samples or 64ms at 16 kHz, which is
% optimal for speech source separation via binary time-frequency masking)
%
% Output:
% X: nfram x nbin x nchan matrix containing the STFT coefficients

%%% Errors and warnings %%%
if nargin<1, error('Not enough input arguments.'); end
if nargin<2, wlen=1024; end
[nsampl,nchan]=size(x);
if nchan>nsampl, error('The signals must be within rows.'); end
if wlen~=4*floor(wlen/4), error('The window length must be a multiple of 4.'); end

%%% Computing STFT coefficients %%%
% Defining sine window
win=sin((.5:wlen-.5)/wlen*pi).';
nfram=floor(nsampl/wlen*2);
% Zero-padding for the last frame
x=[x;zeros((nfram+1)*wlen/2-nsampl,nchan)];

% 窗能量计算方式1：对应istft里面也是这样,这种stft和istft加窗方式是对称的
swin=ones((nfram+1)*wlen/2,1);
% for t=0:nfram-1,
%     swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
% end
% swin=sqrt(swin);
% 以下是由结果推断的，减少了上面这个循环
swin(1:wlen/2,1)=win(1:wlen/2);
swin(nfram*wlen/2+1:end,1)=win(wlen/2+1:wlen);

nbin=wlen/2+1;
X=zeros(nfram,nbin,nchan);
for i=1:nchan,
    for t=0:nfram-1,
        % Framing 方式1
        frame=x(t*wlen/2+1:t*wlen/2+wlen,i).*win./swin(t*wlen/2+1:t*wlen/2+wlen);
        
%         % Framing 方式2
%         frame=x(t*wlen/2+1:t*wlen/2+wlen,i).*win;

        % FFT
        fframe=fft(frame);
        X(t+1,:,i)=fframe(1:nbin);
    end
end

return;