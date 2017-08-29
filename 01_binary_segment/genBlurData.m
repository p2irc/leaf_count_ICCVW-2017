clear all; close all; clc;

% --------------- system input ------------------ %
baseImPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_original/';
rgbPath = 'rgb';
fgPath = 'fg';
outBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/';
gaussFilt_1 = fspecial('gaussian', 5, 1);
gaussFilt_2 = fspecial('gaussian', 9, 2);
% ----------------------------------------------- %

outRgbPath = fullfile(outBasePath, rgbPath);
outFgPath = fullfile(outBasePath, fgPath);

rgbInPath = fullfile(baseImPath, rgbPath);
fgInPath = fullfile(baseImPath, fgPath);
assert(isdir(rgbInPath), 'Input RGB directory does not exist\n %s', rgbInPath);
assert(isdir(fgInPath), 'Input FG directory does not exist\n %s', fgInPath);

% check if directory exists, remove old directory and create new ones
if isdir(outBasePath)
    assert(rmdir(outBasePath, 's'), 'Cannot remove old base directory\n %s', outBasePath);
end
if isdir(outRgbPath)
    assert(rmdir(outRgbPath, 's'), 'Cannot remove old RGB directory\n %s', outRgbPath);
end
if isdir(outFgPath)
    assert(rmdir(outFgPath, 's'), 'Cannot remove old FG directory\n %s', outFgPath);
end
assert(mkdir(outBasePath), 'Cannot create new base directory\n %s', outBasePath);
assert(mkdir(outRgbPath), 'Cannot create new RGB directory\n %s', outRgbPath);
assert(mkdir(outFgPath), 'Cannot create new FG directory\n %s', outFgPath);

numFiles = length(dir(fullfile(rgbInPath, '*.png')));
count = 0;
for i = 1:numFiles
    fprintf('file = %d\n', i);
    im_rgb_org = imread(fullfile(rgbInPath, [num2str(i), '.png']));
    im_fg = imread(fullfile(fgInPath, [num2str(i), '.png'])); % max = 1
    im_fg = (im_fg > 0);
    count = count + 1;
    imwrite(im_rgb_org, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));    
    % use original file to generate 2 blurred and 2 sharpened images
%     im_rgb_blur1 = imfilter(im_rgb_org, gaussFilt_1, 'same', 'conv');
%     im_rgb_blur2 = imfilter(im_rgb_org, gaussFilt_2, 'same', 'conv');
%     im_rgb_sharp1 = imsharpen(im_rgb_org, 'Radius', 3, 'Amount', 1);
%     im_rgb_sharp2 = imsharpen(im_rgb_org, 'Radius', 5, 'Amount', 1);
%     % visualize
%     figure(1); imshow(im_rgb_org);
%     figure(2); imshow(im_rgb_blur1);
%     figure(3); imshow(im_rgb_blur2);
%     figure(4); imshow(im_rgb_sharp1);
%     figure(5); imshow(im_rgb_sharp2);

    % blur 1
    count = count + 1;
    im_rgb_mod = imfilter(im_rgb_org, gaussFilt_1, 'same', 'conv');
    imwrite(im_rgb_mod, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));

    % blur 2
    count = count + 1;
    im_rgb_mod = imfilter(im_rgb_org, gaussFilt_2, 'same', 'conv');
    imwrite(im_rgb_mod, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));
    
    % sharp 1    
    count = count + 1;
    im_rgb_mod = imsharpen(im_rgb_org, 'Radius', 3, 'Amount', 1);
    imwrite(im_rgb_mod, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));
    
    % sharp 2    
    count = count + 1;
    im_rgb_mod = imsharpen(im_rgb_org, 'Radius', 5, 'Amount', 1);
    imwrite(im_rgb_mod, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));
    
    % noise 1
    count = count + 1;
    im_rgb_mod = imnoise(im_rgb_org, 'gaussian', 0, 0.002);
    imwrite(im_rgb_mod, fullfile(outRgbPath, [num2str(count), '.png']));
    imwrite(im_fg, fullfile(outFgPath, [num2str(count), '.png']));
    
end
