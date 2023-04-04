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
Eta = 0.2; % 0.1<ETA<0.4

lstm(dataset,TRAIN,IN,OUT,Bias,Eta);

toc;

function lstm(dataset,row,L,N,bias,eta)

w = weights(2,4);
wForget = w(1,:);
wInput = w(2,:);
wCell = w(3,:);
wOutput = w(4,:);
wY_ = w(5,:);

for r = row
    x = dataset(r,1:L); % input value, x
    Y_ = dataset(r,L+2:L+1+N); % Output value, Y hat

    h = zeros(1,L+1); % hidden state, start from h0 = 0
    c = zeros(1,L+1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Feed- Forward %%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Input Gate %
    neti = wInput(1)*x(index) + wInput(2)*h(index) + wInput(3)*bias;
    i = 1/(1+exp(-neti));

    netc = wCell(1)*x(index) + wCell(2)*h(index) + wCell(3)*bias;
    c_ = exp(2*netc)-1/(exp(2*netc)+1);

    % Forget Gate %
    netf = wForget(1)*x(index)+wForget(2)*h(index)+wForget(3)*bias;
    f = 1/(1+exp(-netf));

    % Memory Cell %
    c(index+1) = i*c_ + f*c(index);

    % Output Gate %
    neto = wOutput(1)*x(index)+wOutput(2)*h(index)+wOutput(3)*bias;
    o = 1/(1+exp(-neto));

    h(index+1) = o*tanh(c(index+1));

    y = (wY_(1)+wY_(2))*h(L+1)+wY_(3)*bias;

end
end

function w = Ws(input)
State = 5;
low = -1/sqrt(input);
up = 1/sqrt(input);

w_1d = low + (up-low).*rand(input*State,1);
w = reshape(w_1d,[State,input]);
end

function w = Us(input)
low = -1/sqrt(input);
up = 1/sqrt(input);

w = low + (up-low).*rand(1,input);
end

function w = weights(layer1,layer2)
low = -1/sqrt(layer1);
up = 1/sqrt(layer1);

w_1d = low + (up-low).*rand((layer1+1)*(layer2+1),1);
w = reshape(w_1d,[layer2+1,layer1+1]);
end