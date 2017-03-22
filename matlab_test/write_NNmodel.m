
% //load model
% 	/* example of the model file: 4 layers (3 hidden), layer size, batchnorm_flag, nonlinear type 
% 	 *
% 	 * (if use batch normalization, there are no bias vectors).
% 	 * 4  129-512 1 1    
% 	 * relu_1[129,512] gamma1[512] beta1[512]
% 	 * 512-512 1 1
% 	 * relu_2[512,512] gamma2[512] beta2[512]
% 	 * 512-512 1 1
% 	 * relu_3[512,512] gamma3[512] beta3[512]
% 	 * 512-258 1 0
% 	 * w4[512,258] gamma4[258] beta4[258]  
% 	 * 
% 	 * (if no batch normalization)
% 	 * 3  129-512 0 1    
% 	 * relu_1[129,512] b1[512]
% 	 * 512-512 0 1
% 	 * ...
% 	 */
clear all

nnet='.\data\FW_model\best.nnet';
h5disp(nnet)

fid=fopen('NNmodel','w');
a=[4.0 129.0 512.0 0.0 1.0];
COUNT=fwrite(fid,a,'float32');
COUNT=fwrite(fid, h5read(nnet,'/relu_1/W')', 'float32');
COUNT=fwrite(fid, h5read(nnet,'/relu_1/b'), 'float32');
a=[512 512 0 1];
COUNT=fwrite(fid,a,'float32');
COUNT=fwrite(fid, h5read(nnet,'/relu_2/W')', 'float32');
COUNT=fwrite(fid, h5read(nnet','/relu_2/b'), 'float32');
a=[512 512 0 1];
COUNT=fwrite(fid,a,'float32');
COUNT=fwrite(fid, h5read(nnet,'/relu_3/W')', 'float32');
COUNT=fwrite(fid, h5read(nnet,'/relu_3/b'), 'float32');
a=[512 258 0 0];
out_layer=[h5read(nnet,'/speech_mask_estimate/W'); h5read(nnet,'/noise_mask_estimate/W')];
out_layer_b=[h5read(nnet,'/speech_mask_estimate/b'); h5read(nnet,'/noise_mask_estimate/b')];
COUNT=fwrite(fid,a,'float32');
COUNT=fwrite(fid, out_layer', 'float32');
COUNT=fwrite(fid, out_layer_b, 'float32');
fclose(fid);


%fid2=fopen('NNmodel','r');
%A=fread(fid2,'float32');
%fclose(fid2);

