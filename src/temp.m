clc;
close all;
clear;

wi1 = [1 2 3 4 5 6];
wi2 = 7;
wi3 = 8;
x1 = [1 2 3 4 5 6];
h0 = 2;
b = -1;
t = [x1 h0 b];
w = [wi1 wi2 wi3];

net = t.*w;
s = sum(net);