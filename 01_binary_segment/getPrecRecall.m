clear all; close all; clc;

th_prec = 0.7; % threshold on precision
th_rec = 0.9; % threshold on recall
th_acc = 0.8; % threshold on accuracy

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/test_binSeg/';
inGtPath = 'fg';
inBinPath = 'bs_sum_plain_nobox';
postPath = {'A1','A2','A3','A4','A5'};
% ---------------------------------------

inGtPath = fullfile(basePath, inGtPath);
inBinPath = fullfile(basePath, inBinPath);

for i = 1:length(postPath)
    tmpInGtPath = fullfile(inGtPath, postPath{i});
    tmpInBinPath = fullfile(inBinPath, postPath{i});
    imgList = dir(fullfile(tmpInBinPath, '*.png'));
    prec_avg = 0;
    rec_avg = 0;
    for j = 1:length(imgList)
        gtFileName = [imgList(j).name(1:end-7), 'fg.png'];
        gt = im2single(imread(fullfile(tmpInGtPath, gtFileName))>0);
        bs = im2single(imread(fullfile(tmpInBinPath, imgList(j).name)));
        true_pos = numel(find(bs==1 & gt==1));
        true_neg = numel(find(bs==0 & gt==0));
        false_pos = numel(find(bs==1 & gt==0));
        false_neg = numel(find(bs==0 & gt==1));
        prec_avg = prec_avg + true_pos/(true_pos + false_pos);
        rec_avg = rec_avg + true_pos/(true_pos + false_neg);
%        accuracy = (true_pos + true_neg)/numel(gt);
    end
    prec_avg = prec_avg / length(imgList);
    rec_avg = rec_avg / length(imgList);
    fprintf('dir = %d, prec = %f, rec = %f\n', i, prec_avg, rec_avg);
end
