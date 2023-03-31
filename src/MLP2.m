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
fprintf("R:%d\n",R);

TRAIN = 1:(0.6*R);
VALIDATION = (0.6*R+1):0.8*R;
TEST = (0.8*R+1):R;
% R = 327680;
% fprintf("%d\n",R);
% TRAIN = 1:(0.7*R);
% VALIDATION = (0.7*R+1):R;
% TEST = (0.85*R+1):R;

L= 6; % 6 inputs
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

w1_epoch = zeros(R,(L+1)*M);
w2_epoch = zeros(R,(M+1)*N);

y2_epoch = zeros(R,N);
% Y_epoch = zeros(R,N);

Y_epoch = pre_dataset(:,C-1:C);

Error = zeros(R,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for r = TRAIN
    x = zeros(1,L);
    for i = 1:L
        x(i) = dataset(r,i); % input value, x
    end

    Y_ = zeros(1,N);
    for i = 1:N
        Y_(i) = dataset(r,L+1+i); % Output value, Y hat
    end
%     Y_epoch(r,:) = Y_;

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
        net2(n) = netCurr + BIAS*w2(M+1,n);
        y2(n) = 1/(1+exp(-net2(n))); % Sigmoid function
    end
    y2_epoch(r,:) = y2;

%     E = (0.5/(0.7*R))*(Y_ - y2).^2; % Error cost function
        E = Y_ - y2;
    for i = 1:2
        Error(r,i) = E(i);
    end

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
            w2(m,n) = ETA*delta_out(n)*y1(m) + w2(m,n);
        end
        w2(M+1,n) = ETA*delta_out(n)*BIAS + w2(M+1,n);
    end

    % Hidden - Input
    for m = 1:M
        for l = 1:L
            w1(l,m) = ETA*delta_hid(m)*x(l) + w1(l,m);
        end
        w1(L+1,m) = ETA*delta_hid(m)*BIAS + w1(L+1,m);
    end

    % add w1, w2 weights into epoch array
    w1_1d = reshape(w1,[1,(L+1)*M]);
    w2_1d = reshape(w2,[1,(M+1)*N]);
    for i=1:(L+1)*M
        w1_epoch(r,i) = w1_1d(i);
    end
    for i = 1:(M+1)*N
        w2_epoch(r,i)=w2_1d(i);
    end
end % end Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% END TRAINING %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% VALIDATION %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for r = VALIDATION
    x = zeros(1,L);
    for i = 1:L
        x(i) = dataset(r,i); % input value, x
    end

    Y_ = zeros(1,N);
    for i = 1:N
        Y_(i) = dataset(r,L+1+i); % Output value, Y hat
    end
%     Y_epoch(r,:) = Y_;
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
        net2(n) = netCurr + BIAS*w2(M+1,n);
        y2(n) = 1/(1+exp(-net2(n))); % Sigmoid function
    end
    y2_epoch(r,:) = y2;

%     E = (0.5/(0.7*R))*(Y_ - y2).^2; % Error cost function
        E = Y_ - y2;
    for i = 1:2
        Error(r,i) = E(i);
    end

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
            w2(m,n) = ETA*delta_out(n)*y1(m) + w2(m,n);
        end
        w2(M+1,n) = ETA*delta_out(n)*BIAS + w2(M+1,n);
    end

    % Hidden - Input
    for m = 1:M
        for l = 1:L
            w1(l,m) = ETA*delta_hid(m)*x(l) + w1(l,m);
        end
        w1(L+1,m) = ETA*delta_hid(m)*BIAS + w1(L+1,m);
    end

    % add w1, w2 weights into epoch array
    w1_1d = reshape(w1,[1,(L+1)*M]);
    w2_1d = reshape(w2,[1,(M+1)*N]);
    for i=1:(L+1)*M
        w1_epoch(r,i) = w1_1d(i);
    end
    for i = 1:(M+1)*N
        w2_epoch(r,i)=w2_1d(i);
    end
end % end Validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% END VALIDATION %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for r = TEST
    x = zeros(1,L);
    for i = 1:L
        x(i) = dataset(r,i); % input value, x
    end

    Y_ = zeros(1,N);
    for i = 1:N
        Y_(i) = dataset(r,L+1+i); % Output value, Y hat
    end
%     Y_epoch(r,:) = Y_;

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
        net2(n) = netCurr + BIAS*w2(M+1,n);
        y2(n) = 1/(1+exp(-net2(n))); % Sigmoid function
    end
    y2_epoch(r,:) = y2;

%     E = (0.5/(0.7*R))*(Y_ - y2).^2; % Error cost function
        E = Y_ - y2;
    for i = 1:2
        Error(r,i) = E(i);
    end

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
            w2(m,n) = ETA*delta_out(n)*y1(m) + w2(m,n);
        end
        w2(M+1,n) = ETA*delta_out(n)*BIAS + w2(M+1,n);
    end

    % Hidden - Input
    for m = 1:M
        for l = 1:L
            w1(l,m) = ETA*delta_hid(m)*x(l) + w1(l,m);
        end
        w1(L+1,m) = ETA*delta_hid(m)*BIAS + w1(L+1,m);
    end

    % add w1, w2 weights into epoch array
    w1_1d = reshape(w1,[1,(L+1)*M]);
    w2_1d = reshape(w2,[1,(M+1)*N]);
    for i=1:(L+1)*M
        w1_epoch(r,i) = w1_1d(i);
    end
    for i = 1:(M+1)*N
        w2_epoch(r,i)=w2_1d(i);
    end
end % end Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% END TESTING %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
plot(Y_epoch);

toc;