function [bit_receive, drop_position, maintain_position] = channel_erasure_CS(bit_data, M, ErasureProb)


st_inter = 1;
L_symble = length(bit_data)/M;
N_symble = L_symble;  % Packet bit length
N_bit = N_symble*M;  % Packet bit length


%% channel erasure setting
ErasureNum = ceil(L_symble*ErasureProb);
Erasure_position = randperm(L_symble);
P = ceil(length(bit_data)/N_bit);
Erasure_mask = zeros(N_symble, P);
Erasure_mask(Erasure_position(1:ErasureNum))=1;
NL_bit = mod(length(bit_data),N_bit);
if (NL_bit~=0)
    data = [bit_data, zeros(1, N_bit-NL_bit)];
else
    data = bit_data;
end
data = reshape(data, [N_bit, P]);

%% Interleaving Encoding
interData = zeros(N_symble, P);
demodData = zeros(N_symble, P);
deinter_bit = zeros(N_bit, P);
interData_bit = zeros(N_bit, P);
drop_position = zeros(N_bit, P);
maintain_position = zeros(N_bit, P);
for i = 1:P-1
    interData_bit(:,i) = randintrlv(data(:,i),st_inter+i); % Interleave.
    interData(:,i) = bit2int(interData_bit(:,i),M);
    %%
    Erasure_mask_temp = Erasure_mask(:,i);
    erasuresVec = zeros(N_symble, 1);
    Erasure_cover = randi([0 2^M-1],N_symble,1);
    demodData(:,i) = interData(:,i).*(1-Erasure_mask_temp)+Erasure_cover.*Erasure_mask_temp;
    erasuresVec(Erasure_mask_temp>0)=255;
    erasuresVec = randdeintrlv(int2bit(erasuresVec,M),st_inter+i); % Deinterleave.
    deinter_bit(:,i) = randdeintrlv(int2bit(demodData(:,i),M),st_inter+i); % Deinterleave.
    drop_position(:,i) = erasuresVec;
    maintain_position(:,i) = 1 - erasuresVec;
end

if (NL_bit~=0)
    interData_bit(1:NL_bit,P) = randintrlv(data(1:NL_bit,P),st_inter+P); % Interleave.
    interData(1:NL_bit/M,P) = bit2int(interData_bit(1:NL_bit,P),M);
    %% P-th package
    Erasure_mask_temp = Erasure_mask(:,P);
    erasuresVec = zeros(N_symble, 1);
    Erasure_cover = randi([0 2^M-1],N_symble,1);
    demodData(:,P) = interData(:,P).*(1-Erasure_mask_temp)+Erasure_cover.*Erasure_mask_temp;
    erasuresVec(Erasure_mask_temp>0)=255;
    erasuresVec = randdeintrlv(int2bit(erasuresVec(1:NL_bit/M),M),st_inter+P); % Deinterleave.
    deinter_bit(1:NL_bit,P) = randdeintrlv(int2bit(demodData(1:NL_bit/M,P),M),st_inter+P); % Deinterleave.
    drop_position(1:NL_bit,P) = erasuresVec;
    maintain_position(1:NL_bit,P) = 1 - erasuresVec;
else
    i = P;
    interData_bit(:,i) = randintrlv(data(:,i),st_inter+i); % Interleave.
    interData(:,i) = bit2int(interData_bit(:,i),M);
    %%
    Erasure_mask_temp = Erasure_mask(:,i);
    erasuresVec = zeros(N_symble, 1);
    Erasure_cover = randi([0 2^M-1],N_symble,1);
    demodData(:,i) = interData(:,i).*(1-Erasure_mask_temp)+Erasure_cover.*Erasure_mask_temp;
    erasuresVec(Erasure_mask_temp>0)=255;
    erasuresVec = randdeintrlv(int2bit(erasuresVec,M),st_inter+i); % Deinterleave.
    deinter_bit(:,i) = randdeintrlv(int2bit(demodData(:,i),M),st_inter+i); % Deinterleave.
    drop_position(:,i) = erasuresVec;
    maintain_position(:,i) = 1 - erasuresVec;
end
%%
drop_position = sort(find(drop_position==1));
maintain_position = sort(find(maintain_position==1));
bit_receive = deinter_bit(1:L_symble*M);



end