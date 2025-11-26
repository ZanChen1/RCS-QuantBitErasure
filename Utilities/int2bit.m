function [out]=int2bit(in,k) 


    
    [m_in,n_in]=size(in);
    m_out=ceil(m_in*k);
    n_out=n_in;
    out=zeros(m_out,n_out);
    for i=1:k
        out(i:k:m_out,1:n_out)=fix(mod(in(1:m_in,1:n_in),2^(k-i+1))/2^(k-i));
    end
    out = out(:)';
    

end