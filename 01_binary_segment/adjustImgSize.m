clear all; close all; clc;

MIN_SIZE = 224;
baseImPath = '/media/aich/DATA/leaf_seg_par';
rgbPath = 'rgb';
fgPath = 'fg';

rgbPath = fullfile(baseImPath, rgbPath);
fgPath = fullfile(baseImPath, fgPath);

numDirs = length(dir(rgbPath)) - 2;

%count = 0;
%parfor (i = 1:numDirs, 5)
for i = 1:numDirs
    tmpRgbPath = fullfile(rgbPath, num2str(i));
    tmpFgPath = fullfile(fgPath, num2str(i));
    imgList = dir(fullfile(tmpFgPath, '*.png'));
    %count = 0;
    for j = 5514:length(imgList)
        fprintf('dir = %d, file = %d\n', i, j);
        gt = imread(fullfile(tmpFgPath, imgList(j).name));
        [r,c] = size(gt);
        if (r < MIN_SIZE) || (c < MIN_SIZE)
            % read rgb image also
            im = imread(fullfile(tmpRgbPath, imgList(j).name));
            % calculate padding
            if r < MIN_SIZE
                r_lb = floor((MIN_SIZE - r)/2)+1;
                r_ub = r_lb + r - 1;
                r_size = MIN_SIZE;
            else
                r_lb = 1;
                r_ub = r;
                r_size = r;
            end
            if c < MIN_SIZE
                c_lb = floor((MIN_SIZE - c)/2)+1;
                c_ub = c_lb + c - 1;
                c_size = MIN_SIZE;
            else
                c_lb = 1;
                c_ub = c;
                c_size = c;
            end       
            im_new = uint8(zeros(r_size, c_size, 3));
            gt_new = false(r_size, c_size);
            im_new(r_lb:r_ub, c_lb:c_ub, :) = im;
            gt_new(r_lb:r_ub, c_lb:c_ub) = gt;
            imwrite(im_new , fullfile(tmpRgbPath, imgList(j).name));
            imwrite(gt_new , fullfile(tmpFgPath, imgList(j).name));
        end
      
    end
end