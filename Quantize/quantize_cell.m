function [par,quantize]=quantize_cell(par,quantize,OMEGA,flag)

    if isfield(par,'y')
        if iscell(par.y)==0
            y=cell(size(OMEGA));
            step(1,1)=1;
            step(1,2)=step(1,1)+length(OMEGA{1})-1;
            for i=2:length(OMEGA)
                step(i,1)=step(i-1,2)+1;
                step(i,2)=step(i-1,2)+length(OMEGA{i});
            end
            for i=1:size(step,1)
                y{i}=par.y(step(i,1):step(i,2));
            end
            par.y=y;
        end
    end

    if flag==1
        par.bin=cell(size(par.y));
        if isempty(quantize.Mu)&&isempty(quantize.Sigm)
            y_temp = cell2mat(par.y);
            quantize.Mu=mean(y_temp(:));
            quantize.Sigm=std(y_temp(:));
        end
%         y_dc=par.y{1}(:)-quantize.Mu;
%         [par.bin{1},quantize]=Quantize(y_dc',quantize,quantize.bit(1),1);        
%         par.bin{1}=par.bin{1}-1;
        for i=1:length(par.y)
            if ~isempty(par.y{i})
                [par.bin{i},quantize]=Quantize(par.y{i}(:)',quantize,quantize.bit(i),1);
                par.bin{i}=par.bin{i}-1;
            end
        end
 
        
    elseif flag==0

        par.dec=cell(size(par.bin));
        bin=cell(size(par.bin));
        for i=1:length(par.bin)
            bin{i}=par.bin{i}+1;
            if ~isempty(bin{i})
                [par.dec{i},quantize]=Quantize(bin{i}(:)',quantize,quantize.bit(i),0);
            end
        end
        %par.dec{1}=par.dec{1}+quantize.Mu;

    end



end