function getBboxImage()
%clear all; close all; clc;

% --------------- region proposal inputs -------------- %
numBoxes = 20; % number of object proposals to detect

edgePath = './edges_dollar/';
tbPath = './toolbox_dollar/';

addpath(genpath(edgePath));
addpath(genpath(tbPath));

%%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = numBoxes;  % max number of boxes to detect
% -------------------------------------------------------- %

% --------------- system inputs -------------- %
SIZE_IMG = 224;
inPath = '/media/aich/DATA/databases/leaf_cvppp2017/lcc_original';
outPath = '/media/aich/DATA/databases/leaf_cvppp2017/lcc_bbox';
rgbPath = 'rgb';
fgPath = 'fg';

inRgbPath = fullfile(inPath, rgbPath);
inFgPath = fullfile(inPath, fgPath);
outRgbPath = fullfile(outPath, rgbPath);
outFgPath = fullfile(outPath, fgPath);

% check if directory exists, remove old directory and create new ones
if isdir(outPath)
    assert(rmdir(outPath, 's'), 'Cannot remove old base directory\n %s', outPath);
end

assert(mkdir(outPath), 'Cannot create new base directory\n %s', outPath);

if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), 'Cannot remove old base directory\n %s', outRgbPath);
end

assert(mkdir(outRgbPath), 'Cannot create new base directory\n %s', outRgbPath);

if isdir(outFgPath)
    assert(rmdir(outFgPath, 's'), 'Cannot remove old base directory\n %s', outFgPath);
end

assert(mkdir(outFgPath), 'Cannot create new base directory\n %s', outFgPath);

%------------------------------------------------------------------%

imgList = dir(fullfile(inRgbPath, '*.png'));
for i = 1:length(imgList)
    fprintf('file = %d\n', i);
    im = im2single(imread(fullfile(inRgbPath, imgList(i).name)));
    gtb = imread(fullfile(inFgPath, imgList(i).name))>0;
    [numRows, numCols, ~] = size(im);
    bb = edgeBoxes(im, model, opts); 
    bb_f = edgeBoxes(flip(im), model, opts); 
    bb_flr = edgeBoxes(fliplr(im), model, opts); 
    bb_fflr = edgeBoxes(flip(fliplr(im)), model, opts); 
    box = getBboxIndices(bb, bb_f, bb_flr, bb_fflr, numRows, numCols);
    im = im(box.rmin:box.rmax, box.cmin:box.cmax, :);
    gtb = gtb(box.rmin:box.rmax, box.cmin:box.cmax);
    [im, gtb] = correctImgSize(im, gtb, SIZE_IMG);
    im = im2uint8(im);
    %imshow(im);
    assert((size(gtb,1)>=SIZE_IMG && size(gtb,2)>=SIZE_IMG), ...
        'Image size is less than minimum limit = %d', SIZE_IMG);
    imwrite(im, fullfile(outRgbPath, imgList(i).name));
    imwrite(gtb, fullfile(outFgPath, imgList(i).name));
end

end

function box = getBboxIndices(bb, bb_f, bb_flr, bb_fflr, numRows, numCols)

    bb = round(bb);
    bb_f = round(bb_f);
    bb_flr = round(bb_flr);
    bb_fflr = round(bb_fflr);
    
    box = struct();
    
    % original and its bottom-up rotated version
    c_min1 = min(bb(:,1));
    c_max1 = max(bb(:,1) + bb(:,3) - 1);
    r_min1 = min(bb(:,2));
    r_max1 = max(bb(:,2) + bb(:,4) - 1);   
    
    c_min1_f = min(bb_f(:,1));
    c_max1_f = max(bb_f(:,1) + bb_f(:,3) - 1);
    r_min1_f = min(bb_f(:,2));
    r_max1_f = max(bb_f(:,2) + bb_f(:,4) - 1);
    
    r_min_01 = max(r_min1, numRows - r_max1_f + 1);
    r_max_01 = min(r_max1, numRows - r_min1_f + 1);
    c_min_01 = max(c_min1, c_min1_f);
    c_max_01 = min(c_max1, c_max1_f);
    
    % fliplr and its bottom-up rotated version
    c_min2 = min(bb_flr(:,1));
    c_max2 = max(bb_flr(:,1) + bb_flr(:,3) - 1);
    r_min2 = min(bb_flr(:,2));
    r_max2 = max(bb_flr(:,2) + bb_flr(:,4) - 1);   
    
    c_min2_f = min(bb_fflr(:,1));
    c_max2_f = max(bb_fflr(:,1) + bb_fflr(:,3) - 1);
    r_min2_f = min(bb_fflr(:,2));
    r_max2_f = max(bb_fflr(:,2) + bb_fflr(:,4) - 1);
    
    r_min_02 = max(r_min2, numRows - r_max2_f + 1);
    r_max_02 = min(r_max2, numRows - r_min2_f + 1);
    c_min_02 = min(c_max2, c_max2_f);
    c_max_02 = max(c_min2, c_min2_f);
    c_min_02 = numCols - c_min_02 + 1;
    c_max_02 = numCols - c_max_02 + 1;
    
    box.rmin = max(r_min_01, r_min_02);
    box.rmax = min(r_max_01, r_max_02);
    box.cmin = max(c_min_01, c_min_02);
    box.cmax = min(c_max_01, c_max_02);
%    box.r_min2 = box.r_min1;
%    box.r_max2 = box.r_max1;
%    box.c_min2 = numCols - box.c_max1 + 1;
%    box.c_max2 = numCols - box.c_min1 + 1;   
    
end

function [im_out, gtb_out] = correctImgSize(im, gtb, size_)
    [r,c,~] = size(im);
    
    if (r >= size_) && (c >= size_)
        im_out = im;
        gtb_out = gtb;
        return;
    end
    
    if (r < size_)
        r_lb = floor((size_ - r)/2) + 1;
        r_ub = r_lb + r - 1;
        r_size = size_;
    else
        r_lb = 1;
        r_ub = r_lb + r - 1;
        r_size = r;
    end
    if (c < size_)
        c_lb = floor((size_ - c)/2) + 1;
        c_ub = c_lb + c - 1;
        c_size = size_;
    else
        c_lb = 1;
        c_ub = c_lb + c - 1;
        c_size = c;
    end    
    
    im_out = single(zeros(r_size, c_size, 3));
    gtb_out = false(r_size, c_size);
    im_out(r_lb:r_ub, c_lb:c_ub, :) = im;
    gtb_out(r_lb:r_ub, c_lb:c_ub) = gtb;

end