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

Y_epoch = pre_dataset(:,C-1:C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neurons = IN-1; % % neurons, Range = 1 to L, Best = 2/3*L+N or L-1

Bias = 1;

E = lstm(dataset,TRAIN,IN,neurons,OUT,Bias);

plot(E);
toc;

function Error =lstm(dataset,row,L,M,N,bias)
W = Ws(L);
wub = W_ubs(L);
wy = Weights(M+1,N);

[~,ro] = size(row);
h = zeros(ro+1,M); % hidden state, start from h0 = 0
c = zeros(ro+1,M);
Error = zeros(ro,2);
for r = row
    x = dataset(r,1:L); % input value, x
    Y_ = dataset(r,L+2:L+1+N); % Output value, Y hat

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Feed- Forward %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for m = 1:M
        hCurr = h(r,m);
        xs = [x hCurr bias];

        % Input Gate 1 %
        wi = W(1,:);
        wubi = wub(1,:);
        w = [wi wubi];
        neti = sum(w.*xs);
        i = sigmoid(neti);
        % Input Gate 2 %
        wc = W(2,:);
        wubc = wub(2,:);
        w = [wc wubc];
        netc = sum(w.*xs);
        c_ = tanh(netc);

        % Forget Gate %
        wf = W(3,:);
        wubf = wub(3,:);
        w = [wf wubf];
        netf = sum(w.*xs);
        f = sigmoid(netf);

        % Output Gate %
        wo =W(4,:);
        wubo = wub(4,:);
        w = [wo wubo];
        neto = sum(w.*xs);
        o = sigmoid(neto);

        % Memory Cell/ Cell State %
        cNext = i.*c_ + f.*c(r,m);
        c(r+1,m) = cNext;

        % Hidden Layer/ State %
        hNext = o*tanh(cNext);
        h(r+1,m) = hNext;
    end
    
    for n=1:N
        for m = 1:M
            y = wy(m,n)*h(r+1,m);
        end
        y(n) = y+bias*wy(M+1,n);
    end
    
    E = (0.5/(ro))*(Y_ - y).^2; % Error cost function
    Error(r,:) = E;
end
end

function y = sigmoid(net)
y = 1/(1+exp(-net));
end

function y = tanh(net)
y = (exp(2*net)-1)/(exp(2*net)+1);
end

function w = Ws(input)
State = 4;
low = -1/sqrt(input);
up = 1/sqrt(input);

w_1d = low + (up-low).*rand(input*State,1);
w = reshape(w_1d,[State,input]);
end

function w = W_ubs(input)
state = 4;
low = -1/sqrt(input);
up = 1/sqrt(input);

w_1d = low + (up-low).*rand(state*2,1);
w = reshape(w_1d,[state,2]);
end

function w = Weights(layer1,layer2)
low = -1/sqrt(layer1);
up = 1/sqrt(layer1);

w_1d = low + (up-low).*rand((layer1)*(layer2),1);
w = reshape(w_1d,[layer1,layer2]);
end