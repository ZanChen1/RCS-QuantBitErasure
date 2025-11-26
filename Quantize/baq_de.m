function [sig_block_de]=baq_de(sig_block_en,y,Mu,Sigm)

[w h]=size(sig_block_en);
sig_block_de=zeros(w,h);
for ii=1:w*h

   sig_block_de(ii)=y(sig_block_en(ii));
  
end
sig_block_de=sig_block_de*Sigm+Mu;
%sig_block_de=round(sig_block_de);

end