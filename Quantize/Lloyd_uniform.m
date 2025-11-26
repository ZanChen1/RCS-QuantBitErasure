function [y,b]=Lloyd_uniform(num)

% Mu=mean(s(:));
% Sigm=std(s(:));
% s_norm=(s-Mu)/Sigm;

%%
b=zeros(1,num+1);
y=zeros(1,num);
% b(1)=min(s_norm);
% b(num+1)=max(s_norm);
b(1)=-3.6;
b(num+1)=3.6;
%% 
delta=(b(num+1)-b(1))/num;
for ii=1:num
    b(ii+1)=b(ii)+delta;
    y(ii)=(b(ii+1)+b(ii))/2;
    
end
b(1)=-inf;
b(num+1)=inf;
savefile=['baq_' num2str(num),'_','Uniform','.mat'];
save(savefile,'y','b');

end