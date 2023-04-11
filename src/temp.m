clc;
close all;
clear;

tic;

% wi1 = [1 2 3 4 5 6];
% wi2 = 7;
% wi3 = 8;
% x1 = [1 2 3 4 5 6];
% h0 = 2;
% b = -1;
% t = [x1 h0 b];
% w = [wi1 wi2 wi3];
%
% net = t.*w;
% s = sum(net);

w1 = [1 2 3 ];
w2 = [2;1];
w = w1.*w2;
display(w);

toc;

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