function genRotBboxData()
restoredefaultpath;

% --------------- system input ------------------ %
rng(72);
baseImPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/';
rgbPath = 'rgb';
fgPath = 'fg';
outBasePath = '/media/aich/DATA/leaf_seg/';
numOutDirs = 90; % put all the images in 90 directories randomly
stepRot = 4; % rotational step size in degree
stepEstimLargeSqr = 5; % step size for largest square estimation
zeroThLargeSqr = 5; % maximum zeros allowed for largest square estimation
minSize = 224; % minimum image size 
% ----------------------------------------------- %

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

for i = 1:numOutDirs
    tmpRgbPath = fullfile(outRgbPath, num2str(i));
    tmpFgPath = fullfile(outFgPath, num2str(i));
    assert(mkdir(tmpRgbPath), 'Cannot create new RGB directory\n %s', tmpRgbPath);
    assert(mkdir(tmpFgPath), 'Cannot create new FG directory\n %s', tmpFgPath);
end

numFiles = length(dir(fullfile(rgbInPath, '*.png')));
count = 0;
numFilesPerDir = single(zeros(numOutDirs, 1));

for i = 1:numFiles
    fprintf('file = %d\n', i);
    count = count + 1;
    % original image
    im_rgb_org = imread(fullfile(rgbInPath, [num2str(i), '.png']));    
    im_fg = (imread(fullfile(fgInPath, [num2str(i), '.png']))); % max = 1
    
    for rot = 0 : stepRot : 359
        fprintf('file = %d, rotation = %d\n', i, rot);
        im_rgb_mod1 = imrotate(im_rgb_org, rot, 'bicubic');
        im_fg1 = imrotate(im_fg, rot, 'bicubic');
        if (rot ~= 0) && (rot ~=180)
            [im_rgb_mod1, im_fg1] = estimLargestSquare(im_rgb_mod1, im_fg1, ...
                        stepEstimLargeSqr, zeroThLargeSqr, minSize);
        end
        if ~isempty(im_rgb_mod1)
            im_rgb_mod2 = fliplr(im_rgb_mod1);
            im_fg2 = fliplr(im_fg1);
            [numRows, numCols, ~] = size(im_rgb_mod1);
            % get bounding boxes for all 4 flipped images
            bboxes1 = edgeBoxes(im_rgb_mod1, model, opts); 
            bboxes1_flip = edgeBoxes(flip(im_rgb_mod1), model, opts); 
            bboxes2 = edgeBoxes(im_rgb_mod2, model, opts); 
            bboxes2_flip = edgeBoxes(flip(im_rgb_mod2), model, opts); 
            % ------------------------------------------- %
            % get combined object boundary
            box = getBoundaries(bboxes1, bboxes1_flip, bboxes2, ...
                            bboxes2_flip, numRows, numCols);
            im_rgb1 = im_rgb_mod1(box.r_min1:box.r_max1, box.c_min1:box.c_max1, :);
            im_rgb2 = im_rgb_mod2(box.r_min2:box.r_max2, box.c_min2:box.c_max2, :);
            im_fg1 = im_fg1(box.r_min1:box.r_max1, box.c_min1:box.c_max1);
            im_fg2 = im_fg2(box.r_min2:box.r_max2, box.c_min2:box.c_max2);

%             figure(1); 
%             subplot(2,2,1); imshow(im_rgb1);
%             subplot(2,2,2); imshow(im_fg1);
%             subplot(2,2,3); imshow(im_rgb2);
%             subplot(2,2,4); imshow(im_fg2);

            %im_fg1 = uint8(im_fg1);
            %im_fg1 = bsxfun(@plus, im_fg1, 1); % needed for SCE training in Torch
            %im_fg2 = uint8(im_fg2);
            %im_fg2 = bsxfun(@plus, im_fg2, 1); % needed for SCE training in Torch           
            
            randDir = randi(numOutDirs); % select a random directory
            numFilesPerDir(randDir) = numFilesPerDir(randDir) + 1;
            imgNum = numFilesPerDir(randDir);
            imwrite(im_rgb1, fullfile(outRgbPath, num2str(randDir), [num2str(imgNum), '.png']));
            imwrite(im_fg1, fullfile(outFgPath, num2str(randDir), [num2str(imgNum), '.png']));  
            randDir = randi(numOutDirs); % select a random directory
            numFilesPerDir(randDir) = numFilesPerDir(randDir) + 1;
            imgNum = numFilesPerDir(randDir);
            imwrite(im_rgb2, fullfile(outRgbPath, num2str(randDir), [num2str(imgNum), '.png']));
            imwrite(im_fg2, fullfile(outFgPath, num2str(randDir), [num2str(imgNum), '.png']));                         
            count = count + 2;
        end
    end
end

disp(['Total number of RGB files written = ', num2str(count)]);
restoredefaultpath;

end

function [box] = getBoundaries(bboxes1, bboxes1_flip, bboxes2, ...
                        bboxes2_flip, numRows, numCols)
    box = struct();
    
    % original and its bottom-up rotated version
    c_min1 = min(bboxes1(:,1));
    c_max1 = max(bboxes1(:,1) + bboxes1(:,3) - 1);
    r_min1 = min(bboxes1(:,2));
    r_max1 = max(bboxes1(:,2) + bboxes1(:,4) - 1);   
    
    c_min1_flip = min(bboxes1_flip(:,1));
    c_max1_flip = max(bboxes1_flip(:,1) + bboxes1_flip(:,3) - 1);
    r_min1_flip = min(bboxes1_flip(:,2));
    r_max1_flip = max(bboxes1_flip(:,2) + bboxes1_flip(:,4) - 1);
    
    r_min_01 = max(r_min1, numRows - r_max1_flip + 1);
    r_max_01 = min(r_max1, numRows - r_min1_flip + 1);
    c_min_01 = max(c_min1, c_min1_flip);
    c_max_01 = min(c_max1, c_max1_flip);
    
    % fliplr and its bottom-up rotated version
    c_min2 = min(bboxes2(:,1));
    c_max2 = max(bboxes2(:,1) + bboxes2(:,3) - 1);
    r_min2 = min(bboxes2(:,2));
    r_max2 = max(bboxes2(:,2) + bboxes2(:,4) - 1);   
    
    c_min2_flip = min(bboxes2_flip(:,1));
    c_max2_flip = max(bboxes2_flip(:,1) + bboxes2_flip(:,3) - 1);
    r_min2_flip = min(bboxes2_flip(:,2));
    r_max2_flip = max(bboxes2_flip(:,2) + bboxes2_flip(:,4) - 1);
    
    r_min_02 = max(r_min2, numRows - r_max2_flip + 1);
    r_max_02 = min(r_max2, numRows - r_min2_flip + 1);
    c_min_02 = min(c_max2, c_max2_flip);
    c_max_02 = max(c_min2, c_min2_flip);
    c_min_02 = numCols - c_min_02 + 1;
    c_max_02 = numCols - c_max_02 + 1;
    
    box.r_min1 = max(r_min_01, r_min_02);
    box.r_max1 = min(r_max_01, r_max_02);
    box.c_min1 = max(c_min_01, c_min_02);
    box.c_max1 = min(c_max_01, c_max_02);
    box.r_min2 = box.r_min1;
    box.r_max2 = box.r_max1;
    box.c_min2 = numCols - box.c_max1 + 1;
    box.c_max2 = numCols - box.c_min1 + 1;    
    
end

function [im, im_fg] = estimLargestSquare(im, im_fg, stepSize, maxZeros, minSize)
    im_gray = rgb2gray(im);
    [r,c,~] = size(im);
    [~, ind1] = find(im(r,:)>0, 1);
    [ind2, ~] = find(im(:,c)>0, 1);
    p1 = [(r+ind2)/2, (c+ind1)/2];
    
    [ind3, ~] = find(im(:,1)>0, 1);
    [~, ind4] = find(im(1,:)>0, 1);
    p2 = [(1+ind3)/2, (1+ind4)/2];
    % estimate the center point in the rotated image
    mid = round([(p1(1)+p2(1))/2, (p1(2)+p2(2))/2]);

    
    if mod(minSize, 2) == 0 
        val_before = minSize/2 - 1; % not local scope, exception
        val_after = minSize/2; % not local scope, exception
    else
        val_before = floor(minSize/2);
        val_after = floor(minSize/2);
    end
    tmpSize = minSize;
    
    while(1)
        if (numel(find(im_gray(mid(1) - val_before, :)>0)) > tmpSize) && ...
                (numel(find(im_gray(mid(1) + val_after, :)>0)) > tmpSize) && ...
                (numel(find(im_gray(:, mid(2) - val_before)>0)) > tmpSize) && ...
                (numel(find(im_gray(:, mid(2) + val_after)>0)) > tmpSize)
            val_before = val_before + stepSize;
            val_after = val_after + stepSize;
            tmpSize = val_before + val_after - maxZeros;
        else
            val_before = val_before - stepSize;
            val_after = val_after - stepSize;
            if val_before + val_after >= minSize
                im = im(mid(1) - val_before : mid(1) + val_after, ...
                        mid(2) - val_before : mid(2) + val_after, :);
                im_fg = im_fg(mid(1) - val_before : mid(1) + val_after, ...
                        mid(2) - val_before : mid(2) + val_after);                    
            else 
                im = [];
                im_fg = [];
            end
            return;
        end
    end
end
