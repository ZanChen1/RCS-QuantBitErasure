function [out]=bit2int(in,k) 

    
    in = reshape(in,[k, length(in)/k]);
    [m_in,n_in]=size(in);
    in=[in;zeros(mod(k-mod(m_in,k),k),n_in)];
    [m_in,n_in]=size(in);
    m_out=ceil(m_in/k);
    n_out=n_in;
    out=zeros(m_out,n_out);
    for i=1:k
        out(1:m_out,1:n_out)=out(1:m_out,1:n_out)+2^(k-i)*in(i:k:m_in,1:n_in);
    end
    

end