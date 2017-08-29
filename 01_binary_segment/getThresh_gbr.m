clear all; close all; clc;

inPath = '../../../databases/leaf_cvppp2017/lcc_original';
rgbPath = 'rgb';
fgPath = 'fg';

rgbPath = fullfile(inPath, rgbPath);
fgPath = fullfile(inPath, fgPath);

imgList = dir(fullfile(rgbPath, '*.png'));

rat_maxmin = single(zeros(length(imgList), 2));

for i = 1:length(imgList)
    fprintf('file = %d\n', i);
    im = im2single(imread(fullfile(rgbPath, imgList(i).name)));
    gtb = imread(fullfile(fgPath, imgList(i).name));
    imr = im(:,:,1);
    img = im(:,:,2);
    imb = im(:,:,3);

    rat_bg_max = max(bsxfun(@rdivide, 2*img(gtb==0), imr(gtb==0) + imb(gtb==0)));
    rat_fg_min = min(bsxfun(@rdivide, 2*img(gtb==1), imr(gtb==1) + imb(gtb==1)));
    rat_maxmin(i,:) = [rat_bg_max, rat_fg_min];
end

save('ratio_grb_maxmin.mat', 'rat_maxmin');