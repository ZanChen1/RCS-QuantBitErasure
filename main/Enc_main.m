function [Trans]=Enc_main(ori_im, measure, quantize)

par.y=measure.A(ori_im(:));
Trans.y_groundtruth = par.y;
[par,quantize]=quantize_cell(par,quantize,measure.OMEGA,1);
for i=1:quantize.layer
    par.bit{i} = int2bit(par.bin{i},quantize.bit(i));   
    Trans.bit_send{i} = par.bit{i};
    Trans.ground_symble{i} = par.bin{i}; 
end
Trans.Mu = quantize.Mu;
Trans.Sigm = quantize.Sigm;
bpp = 0;
for i=1:quantize.layer
    bpp=bpp+length(Trans.bit_send{i});
end
bpp = bpp/measure.image_width/measure.image_height;
fprintf('send bpp=%d; \n',bpp);

end