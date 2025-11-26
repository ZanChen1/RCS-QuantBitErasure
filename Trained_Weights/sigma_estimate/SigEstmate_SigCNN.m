function output=SigEstmate_SigCNN_2(noisy)


P_scale = max(max(max(abs(noisy))));
noisy = single(noisy./(P_scale./2)-1);

noisy=real(noisy);
[height,width] = size(noisy);
noisy = noisy';
noisy = reshape(noisy,1,height*width);
output = double(py.Sigma_hat_matlab.sigma(noisy,height,width)); %
output = double(output*P_scale/2);
end