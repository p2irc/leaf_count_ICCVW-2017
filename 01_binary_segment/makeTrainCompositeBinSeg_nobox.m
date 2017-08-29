clear all; close all; clc;

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/';
rgbPath = 'rgb';
segPath = 'bs_sum_plain_nobox';
comPath = 'comp_bs_sum_plain';
postPath = {'A1','A2','A3','A4'};

rgbPath = fullfile(basePath, rgbPath);
segPath = fullfile(basePath, segPath);
comPath = fullfile(basePath, comPath);

for i = 1:length(postPath)
    tmpRgbPath = fullfile(rgbPath, postPath{i});
    tmpSegPath = fullfile(segPath, postPath{i});
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
        bs = imread(fullfile(tmpSegPath, imgList(j).name));
        im = [im, cat(3,bs,bs,bs)];
        imwrite(im, fullfile(tmpComPath, imgList(j).name));
    end
end
