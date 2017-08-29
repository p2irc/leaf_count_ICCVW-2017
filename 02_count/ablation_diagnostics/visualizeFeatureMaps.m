clear all; close all; clc;

%size_plot = [4, 8];
%size_plot = [8, 8];
size_plot = [8, 16];
%resizeFactor = 4;
resizeFactor = 2;
%resizeFactor = 2;
%load seg_2.mat;
%load seg_4.mat;
load seg_6.mat;
x = permute(x, [2,3,1]);

figure(1);
for i = 1:size(x,3)
    fprintf('%d\n', i);
    tmp = imresize(x(:,:,i), 1/resizeFactor, 'bicubic');
    subplot(size_plot(1), size_plot(2), i); imshow(mat2gray(tmp));    
end

%load noseg_2.mat;
%load noseg_4.mat;
load noseg_6.mat;
x = permute(x, [2,3,1]);

figure(2);
for i = 1:size(x,3)
    fprintf('%d\n', i);
    tmp = imresize(x(:,:,i), 1/resizeFactor, 'bicubic');
    subplot(size_plot(1), size_plot(2), i); imshow(mat2gray(tmp));    
end