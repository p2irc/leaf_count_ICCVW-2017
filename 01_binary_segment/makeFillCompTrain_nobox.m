clear all; close all; clc;

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/';
rgbPath = 'rgb';
segPath = 'bs_sum_plain_nobox';
fillPath = 'fill_bs_sum_plain_nobox';
comPath = 'comp_fill_bs_sum_plain';
postPath = {'A1','A2','A3','A4'};

rgbPath = fullfile(basePath, rgbPath);
segPath = fullfile(basePath, segPath);
fillPath = fullfile(basePath, fillPath);
comPath = fullfile(basePath, comPath);

for i = 1:length(postPath)
    tmpRgbPath = fullfile(rgbPath, postPath{i});
    tmpSegPath = fullfile(segPath, postPath{i});
    tmpFillPath = fullfile(fillPath, postPath{i});
    tmpComPath = fullfile(comPath, postPath{i});
    % remove old directory and create fresh
    if isdir(tmpComPath)
        assert(rmdir(tmpComPath, 's'), ...
            'Cannot remove old composite directory\n %s', tmpComPath);
    end
    assert(mkdir(tmpComPath), ...
        'Cannot create composite subdirectory\n %s', tmpComPath);
    
    imgList = dir(fullfile(tmpRgbPath, '*.png'));
    for j = 1:length(imgList)
        fprintf('dir = %d, file = %d\n', i,j);
        im = imread(fullfile(tmpRgbPath, imgList(j).name));
        bs1 = imread(fullfile(tmpSegPath, imgList(j).name));
        bs2 = imread(fullfile(tmpFillPath, imgList(j).name));
        im = [im, cat(3,bs1,bs1,bs1), cat(3, bs2,bs2,bs2)];
        imwrite(im, fullfile(tmpComPath, imgList(j).name));
    end
end
