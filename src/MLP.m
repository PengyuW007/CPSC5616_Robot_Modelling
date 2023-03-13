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
%%%%%%%%dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Initialization %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[R,C] = size(dataset);
fprintf("R:%d\n",R);

TRAIN = 1:0.7*R;
VALIDATION = (0.7*R+1):0.85*R;
TEST = (0.85*R+1):R;

L = 6; % 6 inputs
N = 2; % 2 outputs
M = L-1; % % neurons, Range = 1 to L, Best = 2/3*L+N or L-1

BIAS = 1;
ETA = 0.25; % 0.1<ETA<0.4


low1 = -1/sqrt(L);
up1 = 1/sqrt(L);
w1_1d = low1 +(up1-low1).*rand((L+1)*M,1);
w1 = reshape(w1_1d,[L+1,M]);

low2 = -1/sqrt(M);
up2 = 1/sqrt(M);
w2_1d = low2 + (up2-low2).*rand((M+1)*N,1);
w2 = reshape(w2_1d,[(M+1),N]);

for r = TRAIN
    x = zeros(1,L);
    for i = 1:L
        x(i) = dataset(r,i); % input value, x
    end

    Y_ = zeros(1,N);
    for i = 1:N
        Y_(i) = dataset(r,L+1+i); % Output value, Y hat
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Feed- Forward %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%
    % Input - Hidden %
    %%%%%%%%%%%%%%%%%%
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

    %%%%%%%%%%%%%%%%%%%
    % Hidden - Output %
    %%%%%%%%%%%%%%%%%%%
    % NET and ACTIVATION (weighted sum) for each neuron I, M - Neurons, N - Output
    net2 = zeros(1,N); % Net
    y2 = zeros(1,N); % Activation function
    netCurr = 0;
    for n = 1:N
        for m = 1:M
            netCurr = netCurr+y1(m)*w2(m,n);
        end
        net2(n) = netCurr + BIAS*w2(M+1,n);
        y2(n) = net2(n);
    end

    E = 0.5*(Y_ - y2).^2; % Error cost function

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Feed- Backward %%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Deltas %
    % Output unit
    delta_out = y2.*(1-y2).*(Y_ - y2);

    % Hidden unit
    delta_hid = zeros(M,1);
    for n = 1:N
        for m = 1:M
            delta_hid(m) = y1(m)*(1 - y1(m))*w2(m,n)*delta_out(n);
        end
    end

    % Updated weights %
    % Output - Hidden
    %w2_new = zeros(M+1,N);
    for n = 1:N
        for m = 1:M
            w2(m,n) = ETA*delta_out(n)*y1(m) + w2(m,n);
        end
        w2(M+1,n) = ETA*delta_out(n)*BIAS + w2(M+1,n);
    end

    % Hidden - Input
    %w1_new = zeros(L+1,M);
    for m = 1:M
        for l = 1:L
            w1(l,m) = ETA*delta_hid(m)*x(l) + w1(l,m);
        end
        w1(L+1,m) = ETA*delta_hid(m)*BIAS + w1(L+1,m);
    end
end % end Training
toc;