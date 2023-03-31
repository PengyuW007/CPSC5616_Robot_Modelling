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
pre_dataset = dataset;
index = randperm(R);
dataset = dataset(index, :);

TRAIN = 1:(0.6*R);
VALIDATION = (0.6*R+1):0.8*R;
TEST = (0.8*R+1):R;

IN= 6; % 6 inputs
OUT = 2; % 2 outputs

% w1_epoch = zeros(R,(IN+1)*NEURONS);
% w2_epoch = zeros(R,(NEURONS+1)*OUT);
%
% y2_epoch = zeros(R,OUT);

Y_epoch = pre_dataset(:,C-1:C);

% Error = zeros(R,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neurons = IN-1; % % neurons, Range = 1 to L, Best = 2/3*L+N or L-1

BIAS = 1;
ETA = 0.2; % 0.1<ETA<0.4

[w1,w1_1d] = weights(IN,neurons);
[w2,w2_1d] = weights(neurons,OUT);

[Error,y2_epoch,w1_epoch,w2_epoch] = bp(dataset,TRAIN,IN,neurons,OUT,BIAS,ETA,w1,w2);
multiplots(Error,TRAIN,w1_epoch,w2_epoch,y2_epoch,Y_epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% VALIDATION %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neurons = 2/3*IN+OUT; % % neurons, Range = 1 to L, Best = 2/3*L+N or L-1

BIAS = 1;
ETA = 0.25; % 0.1<ETA<0.4

[w1,w1_1d] = weights(IN,neurons);
[w2,w2_1d] = weights(neurons,OUT);

[Error,y2_epoch,w1_epoch,w2_epoch] = bp(dataset,VALIDATION,IN,neurons,OUT,BIAS,ETA,w1,w2);
multiplots(Error,VALIDATION,w1_epoch,w2_epoch,y2_epoch,Y_epoch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Error,y2_epoch,w1_epoch,w2_epoch] = bp(dataset,TEST,IN,neurons,OUT,BIAS,ETA,w1,w2);
multiplots(Error,TEST,w1_epoch,w2_epoch,y2_epoch,Y_epoch);

toc;

function [Error,y2_epoch,w1_epoch,w2_epoch]=bp(dataset,row,L,M,N,bias,eta,w1,w2)
for r = row
    x = dataset(r,1:L); % input value, x

    Y_ = dataset(r,L+2:L+1+N); % Output value, Y hat

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
        net1(m) = netCurr + bias*w1(L+1,m);
        y1(m) = 1/(1+exp(-net1(m))); % Sigmoid function
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
        net2(n) = netCurr + bias*w2(M+1,n);
        y2(n) = 1/(1+exp(-net2(n))); % Sigmoid function
    end
    y2_epoch(r,:) = y2;

    len = length(row);
    E = (0.5/(0.7*len))*(Y_ - y2).^2; % Error cost function
    Error(r,:) = E;

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
    for n = 1:N
        for m = 1:M
            w2(m,n) = eta*delta_out(n)*y1(m) + w2(m,n);
        end
        w2(M+1,n) = eta*delta_out(n)*bias + w2(M+1,n);
    end

    % Hidden - Input
    for m = 1:M
        for l = 1:L
            w1(l,m) = eta*delta_hid(m)*x(l) + w1(l,m);
        end
        w1(L+1,m) = eta*delta_hid(m)*bias + w1(L+1,m);
    end

    % add w1, w2 weights into epoch array
    w1_1d = reshape(w1,[1,(L+1)*M]);
    w2_1d = reshape(w2,[1,(M+1)*N]);

    w1_epoch(r,:) = w1_1d;
    w2_epoch(r,:) = w2_1d;
end

end % end BP

function [w,w_1d] = weights(layer1,layer2)
low = -1/sqrt(layer1);
up = 1/sqrt(layer1);

w_1d = low + (up-low).*rand((layer1+1)*layer2,1);
w = reshape(w_1d,[layer1+1,layer2]);
end

function multiplots(Error,row,w1_epoch,w2_epoch,y2_epoch,Y_epoch)
figure;
plot(Error);
xlabel("Iteration");
ylabel("Error cost value");

figure;
subplot(1,2,1);
plot(w1_epoch);
xlabel("Iteration");
ylabel("w1 Weights");
subplot(1,2,2);
plot(w2_epoch);
xlabel("Iteration");
ylabel("w2 Weights");

figure;
plot(y2_epoch);
hold on;
plot(Y_epoch(row,:));
end