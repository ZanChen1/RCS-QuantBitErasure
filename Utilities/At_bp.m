function x = At_bp(b,OMEGA,P_image,P_block,image_height,image_width,block_height,block_width,Phi)


    
    n=length(P_image)/(block_width*block_height);
    [M,N] = size(Phi);
    fx=zeros(n,M);
    if iscell(OMEGA)
        OMEGA=cell2mat(OMEGA);
    end
    if iscell(b)
        b=cell2mat(b);
    end
    fx(OMEGA)=b;
   
    fx=fx';
    x=zeros(N,n);
%     for i=1:n
%         B_temp=reshape(fx(:,i),[block_size,block_size]);
%         B_temp=Phi'*B_temp*Phi;
%         x(:,i)=B_temp(:);
%     end
%     x(P_block,:)=x;
    x(P_block,:)=Phi'*fx;
    x=col2im(x,[block_height,block_width],[image_height,image_width],'distinct');
    %x=col2im(x,[block_size block_size],[320 480],'distinct');
    %x=col2im(x,[block_size block_size],[512 512],'distinct');
    x=x(:);
    x(P_image)=x;
    
end