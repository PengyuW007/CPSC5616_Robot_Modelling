clc;
close all;
clear;

tic;

A = [1 2];
B = [3 4;5 6];

[c,d] = add(A,B);
 

toc;
function [c,d] = add(a,b)
    c = a;
    d = b;
end

