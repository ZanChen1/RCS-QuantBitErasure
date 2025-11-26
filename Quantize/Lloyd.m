function [y,b]=Lloyd(num,distribution)

if strcmp(distribution,'Gaussian')
    F=@(x)1/(sqrt(2*pi)*1)*exp(-(x.^2)/(2*1^2));
    Fx=@(x)1/(sqrt(2*pi)*1)*x.*exp(-(x.^2)/(2*1^2));
end

if strcmp(distribution,'Student')
     
     v=100;
     F=@(x)tpdf(x,v);
     Fx=@(x)tpdf(x,v).*x;
end


%%
y=10*(rand(1,num)-0.5);%产生16个随机数
y=sort(y);
%%
b=zeros(1,num+1);
y2=zeros(1,num);
b(1)=-inf;
b(num+1)=inf;
%% Lloyd max quantizer
while(sum(abs(y-y2))>0.01)
y2=y;
for ii=1:(num-1)
b(ii+1)=(y2(ii)+y2(ii+1))/2;
end
%%
for ii=1:num
y(ii) = quadgk(Fx,b(ii),b(ii+1))/quadgk(F,b(ii),b(ii+1));
end

end
savefile=['baq_' num2str(num),'_',distribution,'.mat'];
save(savefile,'y','b');

end