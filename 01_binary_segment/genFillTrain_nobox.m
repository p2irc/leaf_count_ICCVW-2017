clear all; close all; clc;

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/';
segPath = 'bs_sum_plain_nobox';
fillPath = 'fill_bs_sum_plain_nobox';
postPath = {'A1','A2','A3','A4'};

segPath = fullfile(basePath, segPath);
fillPath = fullfile(basePath, fillPath);

for i = 1:length(postPath)
    tmpSegPath = fullfile(segPath, postPath{i});
    tmpFillPath = fullfile(fillPath, postPath{i});
    % remove old directory and create fresh
    if isdir(tmpFillPath)
        assert(rmdir(tmpFillPath, 's'), ...
            'Cannot remove old composite directory\n %s', tmpFillPath);
    end
    assert(mkdir(tmpFillPath), ...
        'Cannot create composite subdirectory\n %s', tmpFillPath);
    
    imgList = dir(fullfile(tmpSegPath, '*.png'));
    for j = 1:length(imgList)
        fprintf('dir = %d, file = %d\n', i,j);
        bs = imread(fullfile(tmpSegPath, imgList(j).name));
        bs = imfill(bs, 'holes');
        imwrite(bs, fullfile(tmpFillPath, imgList(j).name));
    end
end