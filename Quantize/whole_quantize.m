function [par]=whole_quantize(par,A,quantize)

OMEGA=quantize.OMEGA;
%l=quantize.refine_layer;
par.y=A(par.rim);


[par,~]=quantize_cell(par,quantize,OMEGA,1);


for i=1:length(OMEGA)

    for j=1:length(par.bin{i})
        tau = 0.05;
        t_1 = par.bin{i}(j);
        t_Pvalue = par.Pvalue{i}{j};
        %par.bin{i}(j) = Min_x(t_Pvalue,t_1);
        par.bin{i}(j) = Softmin_x(t_Pvalue,t_1, tau);
        while isnan(par.bin{i}(j))
            tau = tau*0.9;
            par.bin{i}(j) = Softmin_x(t_Pvalue,t_1, tau);
        end

    end

end
%par.bin{3}=par.or_bin{3};
[par,~]=quantize_cell(par,quantize,OMEGA,0);




end

function Out = Softmin_x(In,Tar,k)

lambda = 0.05; % 0.1 or 0.05
In = (lambda*Tar+In)./(lambda+1);
Exp_in = exp(-k*(In-Tar).^2);
if (sum(Exp_in) == 0)
    Exp_in = ones(size(In));
    Out = round((sum(Exp_in.*In))/sum(Exp_in));
else
    Out = round((sum(Exp_in.*In))/sum(Exp_in));
end
%friction_up = (lambda_1*sum(Exp_in.*Tar)+lambda_2*sum(Exp_in.*In));


% friction_up = (sum(Exp_in.*(lambda_1*Tar+lambda_2*In)));
% friction_down = sum(Exp_in.*(lambda_1+lambda_2));
% Out = fix(friction_up/friction_down);



end

function Out = Min_x(t_Pvalue, t_1)
Dis = abs(t_Pvalue - t_1);
[~,IX]=sort(Dis,'ascend');
Out = t_Pvalue(IX(1));
end
