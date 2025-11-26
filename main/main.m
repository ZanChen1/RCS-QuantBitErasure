clear,close all;
iter_seed = 1;
rand('state',iter_seed);
randn('state',iter_seed);

add_path()

Test_set = {'Set11'};
ER = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
SR = [3.2, 8, 20];

measure.AMP_iters = 25;

Test_image_dir = ['../', Test_set{1}, '/'];
foldname = dir(Test_image_dir);
foldname = foldname(3:end);         

index_ER = 6;                        
channel.Erasure_rate = ER(index_ER);

kk = SR(3);                    
measure.denoize_name = 'DPIR';       % 'DPIR' or 'Restormer'

PSNR_all = 0;
SSIM_all = 0;
num_img = length(foldname);

for k = 1:num_img
    measure.Test_set_name = Test_set{1};
    measure.Test_image_dir = fullfile(Test_image_dir, foldname(k).name);
    measure.Image_name     = foldname(k).name;

    quantize = [];
    quantize.bpp = 0.1 * kk;

    [quantize, measure, channel] = qunantification_para(quantize, measure, channel);

    Trans = Enc_main(measure.ori_im, measure, quantize);

    for i = 1:quantize.layer
        [Trans.bit_receive{i}, Trans.drop_position{i}, Trans.maintain_position{i}] = ...
            channel.channel_erasure_simulate(Trans.bit_send{i});
    end

    [rec_im, measure] = Dec_main(Trans, measure, quantize);
    ori_im = measure.ori_im;

    psnr_now = csnr(rec_im, ori_im, 8, 0, 0);
    ssim_now = cal_ssim(rec_im, ori_im, 0, 0);

    PSNR_all = PSNR_all + psnr_now;
    SSIM_all = SSIM_all + ssim_now;

    fprintf('SR: %f, ER: %f, Image_name:%s, Denoiser:%s, Bpp:%f, seeds:%d, PSNR:%f, SSIM:%f \n', ...
        kk/80, ER(index_ER), measure.Image_name, measure.denoize_name, ...
        quantize.bpp, iter_seed, psnr_now, ssim_now);
end

mean_PSNR = PSNR_all / num_img;
mean_SSIM = SSIM_all / num_img;

fprintf('SR: %f, ER: %f, mean_PSNR: %f, mean_SSIM: %f\n', ...
    kk/80, ER(index_ER), mean_PSNR, mean_SSIM);
