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
index = randperm(R);
dataset = dataset(index, :);

TRAIN = 1:(0.6*R);
VALIDATION = (0.6*R+1):0.8*R;
TEST = (0.8*R+1):R;

L= 6; % 6 inputs
N = 2; % 2 outputs
M = L-1; % % neurons, Range = 1 to L, Best = 2/3*L+N or L-1

BIAS = 1;
ETA = 0.25; % 0.1<ETA<0.4

[w1,w1_1d] = weights(L,M);

toc;

function [w,w_1d] = weights(layer1,layer2)
    low = -1/sqrt(layer1);
    up = 1/sqrt(layer1);

    w_1d = low + (up-low).*rand((layer1+1)*layer2,1);
    w = reshape(w_1d,[layer1+1,layer2]);
end