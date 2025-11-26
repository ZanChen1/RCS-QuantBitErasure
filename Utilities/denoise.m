function [ denoised, output_partial ] = denoise(noisy,sigma_hat,height,width,denoiser)

noisy=reshape(noisy,[height,width]);
output_partial = zeros(height,width);

switch denoiser



    case 'DPIR'
        if sigma_hat>=150
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.MWCNN_matlab.denoise17(noisy,height,width,sigma_hat)); % 
            output = reshape(output,width,height);
            output = output';
        else
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.DPIR_matlab.denoiser(noisy, height, width, sigma_hat));
            output = reshape(output,width,height);
            output = output';
        end


    case 'Restormer'
        if sigma_hat>=150
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.MWCNN_matlab.denoise17(noisy,height,width,sigma_hat)); 
            output = reshape(output,width,height);
            output = output';
        else
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.Restormer_denoise_matlab.denoiser(noisy, height, width, sigma_hat));
            output = reshape(output,width,height);
            output = output';
        end

    otherwise
        error('Unrecognized Denoiser')
end
denoised=output(:);
output_partial = output_partial(:);
end

