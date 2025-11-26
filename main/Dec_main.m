function [Rec_im, measure]=Dec_main(Trans, measure, quantize)

quantize.Mu=Trans.Mu;
quantize.Sigm=Trans.Sigm;
%%
for i=1:quantize.layer
    [Pvalue{i},Pinit{i}] = Bit2PossibleInt(Trans.bit_receive{i}, length(measure.OMEGA{i}), quantize.bit(i), Trans.drop_position{i}, Trans.maintain_position{i});
end
par.bin = Pinit;
par.Pvalue = Pvalue;
quantize.OMEGA = measure.OMEGA;
[par,quantize]=quantize_cell(par,quantize,measure.OMEGA,0);
measure.rate=length(cell2mat(par.dec))/quantize.image_width/quantize.image_height;
y = par.dec;
AMP_iters = measure.AMP_iters; %AMP重建迭代次数
errfxn = @(x_hat) PSNR(abs(measure.ori_im),reshape(abs(x_hat),[measure.image_height,measure.image_width]));
[Rec_im]  = Prox_Moment_Whole(y,AMP_iters,measure.denoize_name,measure,quantize,par,errfxn);

end

