function [quantize, measure, channel] = qunantification_para(quantize, measure, channel)

%%Quantization Parameter Setting
quantize.layer = 1;
quantize.bit=[8];%量化位数
quantize.Rate_proportion=[1];
quantize.distr='Gaussian';    %'Gaussian'or'Uniform' 量化方式
quantize.method='BAQ';%  量化码书生成方法

%%

channel.channel_erasure_simulate = @(bit_send) channel_erasure_CS(bit_send, quantize.bit, channel.Erasure_rate); %按照bit丢



%%
measure.ori_im = double(imread(measure.Test_image_dir));
[quantize.image_height,quantize.image_width]=size(measure.ori_im);
quantize.Mu=[];
quantize.Sigm=[];
bpp = quantize.bpp;
Rate_proportion=quantize.Rate_proportion;
for i=1:quantize.layer
    quantize.rate_allocation(i)=round(quantize.image_width*quantize.image_height*bpp/quantize.bit(i)*Rate_proportion(i));

end
[quantize.prograssive_step]=OMEGA_slip(quantize.rate_allocation);

%%
measure.prograssive_step = quantize.prograssive_step;
measure.rate_allocation = quantize.rate_allocation;
quantize.image_name = measure.Image_name(1:end-4);


%%Measurement Parameter Setting
measure.model = 'Bernoulli';   %% 采用的观测矩阵：Hadarmad or Bernoulli or Diffraction or Cartesian
%%

measure.block_width = 64;
measure.block_height = 64;
[image_height, image_width]=size(measure.ori_im);
measure.image_height = image_height;
measure.image_width = image_width; %divide the whole image into small blocks

%%
q=1:(measure.image_width*measure.image_height);
step=measure.prograssive_step;
for i=1:size(step,1)
    if step(i,1) == 0
        measure.OMEGA{i} = [];
    else
        measure.OMEGA{i}= q(step(i,1):step(i,2));
    end
end
%%
measure.P_image=randperm(measure.image_height*measure.image_width);
measure.P_block=randperm(measure.block_height*measure.block_width);
measure.length = measure.image_width*measure.image_height;
[A, At, measure] = Measure_matrix_create(measure);
measure.A = A;
measure.At = At;

end