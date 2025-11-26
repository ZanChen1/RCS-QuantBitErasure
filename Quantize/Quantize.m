function [s_out,quantize]=Quantize(s,quantize,bit,flag)


q_l=round(2^bit);
distr=quantize.distr;
method=quantize.method;
if strcmp(method,'BAQ')
    
if strcmp(distr,'Gaussian')||strcmp(distr,'Student')
        filename=['baq_',num2str(q_l),'_',distr,'.mat'];
        if exist(filename,'file')
            load(filename);
        else
            [y,b]=Lloyd(q_l,distr);
        end
        
    elseif strcmp(distr,'Uniform')
        filename=['baq_',num2str(q_l),'_',distr,'.mat'];
        if exist(filename,'file')
            load(filename);
        else
            [y,b]=Lloyd_uniform(q_l);
        end
    end
end



%%
s_code=zeros(size(s));
s_q=zeros(size(s));
siz=length(s);
num=fix(length(s)/siz);
Mu=quantize.Mu;
Sigm=quantize.Sigm;
if flag==1
    
    for i=1:num
        
        [s_code((i-1)*siz+1:i*siz)]=baq_en(s((i-1)*siz+1:i*siz),y,b,Mu,Sigm);
        
    end
    
    if (length(s)>num*siz)
        
        [s_code(num*siz+1:end)]=baq_en(s(num*siz+1:end),y,b,Mu,Sigm);
        
    end
    s_out=s_code;
    
end


if flag==0
    
    for i=1:num
        
        
        [s_q((i-1)*siz+1:i*siz)]=baq_de(s((i-1)*siz+1:i*siz),y,Mu(i),Sigm(i));
        
    end
    
    if (length(s)>num*siz)
        
        [s_q(num*siz+1:end)]=baq_de(s(num*siz+1:end),y,Mu(num+1),Sigm(num+1));
    end
    s_out=s_q;
end
%%
%         quantization_error=sum((s_q-s).^2)/length(s_q);
%         y_huff= mat2huff(s_code);
%         bits = imratio(y_huff)*8;
%         rate=bits/256/256;
%%
%         [s_code,error_rate]=join_erro(s_code,q_l,0);
%         [s_q]=baq_de(s_code,y,Mu,Sigm);
%         noise_error=sum((s_q-s).^2)/length(s_q);

end