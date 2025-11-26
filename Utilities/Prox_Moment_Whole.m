function [x_hat] = Prox_Moment_Whole(y_all, AMP_iters, denoiser, measure, quantize, par, PSNR_func)

%% Settings
epsilon = 1;
lambda_1 = 100;
alpha = 1;
%%
y_all = cell2mat(y_all)';
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure.image_height,measure.image_width,denoiser);
OMEGA=cell2mat(measure.OMEGA(1:quantize.layer));
quantize.OMEGA = measure.OMEGA(1:quantize.layer);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi_mt);
if iscell(y_all)
    y=cell2mat(y_all)';
else
    y = y_all;
end

%%
n=measure.length;
m=length(y);
x_t{2} = zeros(n,1);

%% First iteration
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));
x_t{2}=double(denoi(x_t{1},sigma_hat1));
%%
i = 1;
v_t = zeros(n,1);
while i<=AMP_iters
    eta=randn(1,n)/sqrt(m);
    gamma=1/(epsilon)*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    par.rim=x_t{2};
    [par]=whole_quantize(par,M, quantize);
    if iscell(par.dec)
        y=(M(x_t{2})'+lambda_1*cell2mat(par.dec)')/(1+lambda_1);
    end
    i = i+1;
end
x_hat = x_t{2};
x_hat=reshape(x_hat,[measure.image_height,measure.image_width]);
end


