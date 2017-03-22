% offline process

clear all
clc
filename='dibao\jingzhi75dB';
% 此处 n.wav 仅用来做VAD的判断
[n fs]=wavread([filename '_N.wav']);
% 多通道带噪文件
filein=[filename '.wav'];
% processed output
out=zeros(length(n),1);

nfft=256; % half overlap
nbin=nfft/2+1; 
beta=5;
% 离线方式处理：依赖语音VAD，利用语音前面的噪声段计算更新Rnn
% 判断噪声段，取出处理区间
start=1;
ending=0;
flag=0; % 标识语音段
nflag=0; %标识噪声段结尾
while (ending<length(n)-nfft)
    ending=ending+nfft; % one frame
    
    ntmp=n(ending-nfft+1:ending);
    npower=sum(ntmp.^2);
    if npower==0 && nflag==0    %噪声能量为0，噪声结束，语音段开始
        flag=1;     
        nflag=1;
        % 记录噪声段长度，如果太短计算Rnn时考虑平滑上一次的结果
        nblock=wavread(filein, [start ending-nfft]);
    end
    if flag==1 && npower>0  %语音段结束，噪声帧出现，取出该段进行处理
        flag=0;
        nflag=0;
        yblock=wavread(filein, [start ending-nfft]);
        if ending>0
            % processing
            Nblock=stft_multi_2(nblock, nfft);
            Yblock=stft_multi_2(yblock, nfft);
            % calc average Rnn Ryy
            Rnn=average_psd(Nblock);
            if size(Nblock,1)<10
                Rnn=0.9*Rnn_old+0.1*Rnn;
            end
            Rnn_old=Rnn;
            Ryy=average_psd(Yblock);
            % PMWF
            Y_pmwf=process_pmwf_0620(Rnn, Ryy, Yblock, beta);
            %
            out(start:ending-nfft)=istft_multi_2(Y_pmwf, size(yblock,1));
            
%             out(start:ending-nfft)=out(start:ending-nfft)/max(out(start:ending-nfft));
            % end process
        end
        start=ending-nfft+1;
    end
end

% output
wavwrite(out,fs,[filename '_p' int2str(beta) '.wav']);



