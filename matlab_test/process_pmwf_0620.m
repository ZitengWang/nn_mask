function Y_pmwf=process_pmwf_0620(Rnn, Ryy, Y, beta)

[nframe nbin nchan]=size(Y);
% reference channel
ref=1;
u=zeros(nchan,1);
u(ref,ref)=1;

% beta

W=zeros(nchan, nbin); 
tmpy=zeros(nchan, nchan); tmpn=zeros(nchan,nchan);
for bin =1:nbin
    tmpy(:,:)=Ryy(bin,:,:);tmpn(:,:)=Rnn(bin,:,:);
    lamda=trace(tmpn\tmpy)-nchan;
    if lamda<0
        lamda=eps;
    end
     W(:,bin)=tmpn\tmpy*u/(beta+lamda);
    
%     % max SNR
%     beta_plus_lamda=sqrt(Ryy(ref,ref)*lamda);
%     if beta_plus_lamda<1
%         beta_plus_lamda=1;
%     end
%     W(:,bin)=tmpn\tmpy*u/(beta_plus_lamda);
end
% apply the beamformer
Y_pmwf=zeros(nframe,nbin);
for frm=1:nframe
    for bin=1:nbin
        tmp(:,1)=Y(frm,bin,:); Y_pmwf(frm,bin)=W(:,bin)'*tmp;
    end
end