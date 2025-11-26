function [drop_position, maintain_position] = Dropbit_mask(l, k, drop_rate, dropout_model)


drop_length = round(l*k*drop_rate);

switch dropout_model
    case 'random'
        temp_position = randperm(l*k);
        drop_position = sort(temp_position(1:drop_length));
        maintain_position = sort(temp_position(drop_length+1:end));
        
    case 'significant'
        temp_position = reshape([1:l*k],k,l)';
        drop_position = sort(temp_position(1:drop_length));
        maintain_position = sort(temp_position(drop_length+1:end));
        
    case 'two_side'
        
        P = 2/3;
        drop_length_UP = round(drop_length*P);
        drop_length_DOWN = drop_length-drop_length_UP;
        
        temp_position = reshape([1:l*k],k,l)';
        drop_position_UP = sort(temp_position(1:drop_length_UP));
        drop_position_DOWN = sort(temp_position(l*k-drop_length_DOWN+1:end));
        drop_position = [drop_position_UP, drop_position_DOWN];        
        maintain_position = sort(temp_position(drop_length_UP+1:l*k-drop_length_DOWN));
        
    case 'insignificant'
        temp_position = reshape([1:l*k],k,l)';
        drop_position = sort(temp_position(l*k-drop_length+1:end));
        maintain_position = sort(temp_position(1:l*k-drop_length));
        
    otherwise
        error('Unrecognized dropout model')
end

end