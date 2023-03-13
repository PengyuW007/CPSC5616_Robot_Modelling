clc;
close all;
clear;

tic;

file_1 = "Dataset_with_6 inputs and 2 Outputs.xlsx";
file_2 = "Dataset_5000.xlsx";
file_3 = "Dataset_300000.xlsx";

file = file_2;
if (file==file_1)
    dataset = readmatrix(file_1);
elseif(file == file_2)
    dataset = readmatrix(file_2);
elseif(file == file_3)
    dataset = readmatrix(file_3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Initialization %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[R,C] = size(dataset);
fprintf("R:%d\n",R);

L = 6; % 6 inputs
N = 2; % 2 outputs

% neurons
M = L-1; % Range = 1 to L, Best = 2/3*L+N or L-1

BIAS = 1;

low1 = -1/sqrt(L);
up1 = 1/sqrt(L);
w1_1d = low1 +(up1-low1).*rand((L+1)*M,1);
w1 = reshape(w1_1d,[L+1,M]);

low2 = -1/sqrt(M);
up2 = 1/sqrt(M);
w2_1d = low2 + (up2-low2).*rand((M+1)*N,1);
w2 = reshape(w2_1d,[(M+1),N]);


x = zeros(1,L);
for i = 1:L
    x(i) = dataset(1,i); % input value, x
end

Y_ = zeros(1,N);
for i = 1:N
    Y_(i) = dataset(1,L+1+i); % Output value, Y hat
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Feed- Forward %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NET and ACTIVATION (weighted sum) for each neuron I, L - input, M - Neurons
net1 = zeros(1,M); % Net
y1 = zeros(1,M); % Activation function
netCurr = 0;
for m = 1:M
    for l = 1:L
        netCurr = netCurr+x(l)*w1(l,m);
    end
    net1(m) = netCurr + BIAS*w1(L+1,m);
    y1(m) = 1/(1+exp(-net1(m)));
end

toc;