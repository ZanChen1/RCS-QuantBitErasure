function [Pvalue, Pinit] = Bit2PossibleInt(bit, m, k, drop, maintain)

Int = ones(k,m);
Int(maintain) = bit(maintain);
Int(drop) = NaN;
% P_start = 0;
% P_end =0;

for i = 1:m
    
    temp = Int(:,i); 
    temp_value = 0;
    for j = 1:k
        if isnan(temp(j))
            temp_value1 = temp_value;
            temp_value2 = temp_value+2^(k-j);
            temp_value = [temp_value1, temp_value2];
        else
            temp_value = temp_value+2^(k-j)*temp(j);
        end        
    end    
    Pvalue{i} = sort(temp_value);
    Pinit(i) = uint8(min(temp_value)); %%% max for DPIR and min for Restormer
    %Pinit(i) = temp_value(randi(numel(temp_value)));
end
% Pvalue.item = cell2mat(Pvalue.item);

end