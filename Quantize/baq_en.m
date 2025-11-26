function [sig_block_en]=baq_en(sig_block,y,b,Mu,Sigm)

[w h]=size(sig_block);
sig_block_norm=(sig_block-Mu)/Sigm;
sig_block_en=zeros(w,h);
sig_block_de=zeros(w,h);
for ii=1:w*h
    jj=1;
   while(sig_block_norm(ii)>=b(jj))
     jj=jj+1;
   end
   sig_block_en(ii)=jj-1;
   sig_block_de(ii)=y(sig_block_en(ii));
end


end