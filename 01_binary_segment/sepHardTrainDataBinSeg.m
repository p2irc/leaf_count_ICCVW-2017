clear all; close all; clc;

th_prec = 0.7; % threshold on precision
th_rec = 0.9; % threshold on recall
th_acc = 0.8; % threshold on accuracy

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/';
inRgbPath = 'rgb';
inGtPath = 'fg';
inBinPath = 'bs_sum_plain_nobox';
outRgbPath = 'rgb_hard';
outGtPath = 'fg_hard';
postPath = {'A1','A2','A3','A4'};
% ---------------------------------------

inRgbPath = fullfile(basePath, inRgbPath);
inGtPath = fullfile(basePath, inGtPath);
inBinPath = fullfile(basePath, inBinPath);
outRgbPath = fullfile(basePath, outRgbPath);
outGtPath = fullfile(basePath, outGtPath);

% check if directory exists, remove old directory and create new ones
if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), 'Cannot remove old RGB directory\n %s', outRgbPath);
end
if isdir(outGtPath)
    assert(rmdir(outGtPath, 's'), 'Cannot remove old FG directory\n %s', outGtPath);
end
assert(mkdir(outRgbPath), 'Cannot create new RGB directory\n %s', outRgbPath);
assert(mkdir(outGtPath), 'Cannot create new FG directory\n %s', outGtPath);

% create sub directories
for i = 1:length(postPath)
    tmpOutRgbPath = fullfile(outRgbPath, postPath{i});
    tmpOutGtPath = fullfile(outGtPath, postPath{i});    
    if isdir(tmpOutRgbPath)
        assert(rmdir(tmpOutRgbPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutRgbPath);
    end
    if isdir(tmpOutGtPath)
        assert(rmdir(tmpOutGtPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutGtPath);
    end    
    assert(mkdir(tmpOutRgbPath), ...
        'Cannot create RGB subdirectory\n %s', tmpOutRgbPath);
    assert(mkdir(tmpOutGtPath), ...
        'Cannot create GT subdirectory\n %s', tmpOutGtPath);
end
% ----------------------------------------------------------------------


count = 0;
for i = 1:length(postPath)
    tmpInRgbPath = fullfile(inRgbPath, postPath{i});
    tmpInGtPath = fullfile(inGtPath, postPath{i});
    tmpInBinPath = fullfile(inBinPath, postPath{i});
    tmpOutRgbPath = fullfile(outRgbPath, postPath{i});
    tmpOutGtPath = fullfile(outGtPath, postPath{i});
    imgList = dir(fullfile(tmpInRgbPath, '*.png'));
    for j = 1:length(imgList)
        gtFileName = [imgList(j).name(1:end-7), 'fg.png'];
        gt = im2single(imread(fullfile(tmpInGtPath, gtFileName))>0);
        bs = im2single(imread(fullfile(tmpInBinPath, imgList(j).name)));
        true_pos = numel(find(bs==1 & gt==1));
        true_neg = numel(find(bs==0 & gt==0));
        false_pos = numel(find(bs==1 & gt==0));
        false_neg = numel(find(bs==0 & gt==1));
        precision = true_pos/(true_pos + false_pos);
        recall = true_pos/(true_pos + false_neg);
        accuracy = (true_pos + true_neg)/numel(gt);
        if (precision <= th_prec) || (recall <= th_rec) ...
                || (accuracy <= th_acc)
            count = count + 1;
            fprintf('dir = %s, file = %s, prec = %f, rec = %f, acc = %f\n', ...
                postPath{i}, imgList(j).name, precision, recall, accuracy);
            
            % copy files into RGB directory
            assert(copyfile(fullfile(tmpInRgbPath, imgList(j).name), tmpOutRgbPath), ...
                'Cannot copy RGB file = %s', fullfile(tmpInRgbPath, imgList(j).name));
            
            % copy files into GT directory
            assert(copyfile(fullfile(tmpInGtPath, gtFileName), tmpOutGtPath), ...
                'Cannot copy GT file = %s', fullfile(tmpInGtPath, gtFileName));
        end
    end
end

fprintf('Number of low PRA files = %d\n', count);