clear all; close all; clc;

SIZE_MAX = 448;
basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_count';
inCsvPath = basePath;
inRgbPath = 'rgb';
inBinaryPath = 'bs';
outRgbPath = 'rgb_resize';
outBinaryPath = 'bs_resize';
postPath = {'A1','A2','A3','A4'};

inRgbPath = fullfile(basePath, inRgbPath);
inBinaryPath = fullfile(basePath, inBinaryPath);
outRgbPath = fullfile(basePath, outRgbPath);
outBinaryPath = fullfile(basePath, outBinaryPath);

% remove old augmented directory with subdirectories
if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), ...
        'Cannot remove old RGB resize directory\n %s', outRgbPath);
end
assert(mkdir(outRgbPath), 'Cannot create new RGB resize directory\n %s', outRgbPath);
if isdir(outBinaryPath)
    assert(rmdir(outBinaryPath, 's'), ...
        'Cannot remove old BS resize\n %s', outBinaryPath);
end
assert(mkdir(outBinaryPath), 'Cannot create new BS resize directory\n %s', outBinaryPath);

for i = 1:length(postPath)
    tmpInRgbPath = fullfile(inRgbPath, postPath{i});
    tmpInBinaryPath = fullfile(inBinaryPath, postPath{i});
    tmpOutRgbPath = fullfile(outRgbPath, postPath{i});
    tmpOutBinaryPath = fullfile(outBinaryPath, postPath{i});
    assert(mkdir(tmpOutRgbPath), ...
        'Cannot create directory\n %s', tmpOutRgbPath);
    assert(mkdir(tmpOutBinaryPath), ...
        'Cannot create directory\n %s', tmpOutBinaryPath);    
    imgList = dir(fullfile(tmpInRgbPath, '*.png'));
    for j = 1:length(imgList)
        fprintf('dir = %d, file = %d\n', i, j);
        im = imread(fullfile(tmpInRgbPath, imgList(j).name));
        bs = imread(fullfile(tmpInBinaryPath, imgList(j).name));
        rat = SIZE_MAX/max(size(im));
        if rat < 1
            im = imresize(im, rat, 'bicubic');
            bs = imresize(bs, rat, 'bicubic');
        end
        [r,c,~] = size(im);
        if r < SIZE_MAX
            r_lb = floor((SIZE_MAX - r)/2);
            r_ub = r_lb + r - 1;
        else
            r_lb = 1;
            r_ub = SIZE_MAX;
        end
        if c < SIZE_MAX
            c_lb = floor((SIZE_MAX - c)/2);
            c_ub = c_lb + c - 1;
        else
            c_lb = 1;
            c_ub = SIZE_MAX;            
        end
        
        im_new = uint8(zeros(SIZE_MAX, SIZE_MAX, 3));
        bs_new = uint8(zeros(SIZE_MAX));
        im_new(r_lb:r_ub, c_lb:c_ub, :) = im;
        bs_new(r_lb:r_ub, c_lb:c_ub) = bs;
        imwrite(im_new, fullfile(tmpOutRgbPath, imgList(j).name));
        imwrite(bs_new, fullfile(tmpOutBinaryPath, imgList(j).name));
    end
end
