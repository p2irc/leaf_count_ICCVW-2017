function genTrainDataBinSeg_nobox()
%clear all; close all; clc;

restoredefaultpath;

% % % --------------- region proposal inputs -------------- %
% % numBoxes = 20; % number of object proposals to detect
% % 
% % edgePath = './edges_dollar/';
% % tbPath = './toolbox_dollar/';
% %
% % addpath(genpath(edgePath));
% % addpath(genpath(tbPath));
% % 
% % %%% load pre-trained edge detection model and set opts (see edgesDemo.m)
% % model=load('models/forest/modelBsds'); model=model.model;
% % model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
% % 
% % %%% set up opts for edgeBoxes (see edgeBoxes.m)
% % opts = edgeBoxes;
% % opts.alpha = .65;     % step size of sliding window search
% % opts.beta  = .75;     % nms threshold for object proposals
% % opts.minScore = .01;  % min score of boxes to detect
% % opts.maxBoxes = numBoxes;  % max number of boxes to detect
% % % -------------------------------------------------------- %

% -------------------- system input ---------------------- %
SIZE_IMG = 224;
SIZE_STEP = 150; % step size to generate image from the original
inRgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/rgb';
postImPath = {'A1','A2','A3','A4'};
outRgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/rgb_224_nobox';

% create sub directories
for i = 1:length(postImPath)
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    if isdir(tmpOutRgbPath)
        assert(rmdir(tmpOutRgbPath, 's'), ...
            'Cannot remove old RGB directory\n %s', tmpOutRgbPath);
    end

    assert(mkdir(tmpOutRgbPath), ...
        'Cannot create RGB subdirectory\n %s', tmpOutRgbPath);
end
% ----------------------------------------------------------------------

for i = 1:length(postImPath) % subdirectory index
    tmpInRgbPath = fullfile(inRgbPath, postImPath{i});
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    imgList = dir(fullfile(tmpInRgbPath, '*.png'));
    specStruct = struct(); % data specification
    specStruct.fileName = [];
    specStruct.fileSize = [];
    specStruct.subfiles = {};
    specStruct.globalIndices = [];
    specStruct.localIndices = [];
    for j = 1:length(imgList) % file inside subdir index
% %         if i==3 && j==14
% %             disp('debug');
% %         end
        fileName = imgList(j).name;
        im = imread(fullfile(tmpInRgbPath, fileName));
        % extract bounding boxes for all 4 types
        [numRows, numCols, ~] = size(im);
        specStruct(j).fileSize = [numRows, numCols];
%        assert((numRows >= SIZE_IMG && numCols >= SIZE_IMG), ...
%            'Original image size (%d, %d) is less than minimum size.', ...
%            numRows, numCols);
% %         bb = edgeBoxes(im, model, opts); 
% %         bb_f = edgeBoxes(flip(im), model, opts); 
% %         bb_flr = edgeBoxes(fliplr(im), model, opts); 
% %         bb_fflr = edgeBoxes(flip(fliplr(im)), model, opts); 
% %         box = getBboxIndices(bb, bb_f, bb_flr, bb_fflr, numRows, numCols);
% % %        imshow(im(box.rmin:box.rmax, box.cmin:box.cmax, :));
% %         im = im(box.rmin:box.rmax, box.cmin:box.cmax, :);
% %         % add this offsets to the extract indices value while merging
% %         rowOffset = box.rmin - 1;
% %         colOffset = box.cmin - 1;
        rowOffset = 0;
        colOffset = 0;
        [im, localIndices] = correctImgSize(im, SIZE_IMG);
%        imshow(im);
        % crop and save images
        [numRows, numCols, ~] = size(im);
        row_st = 1; col_st = 1; 
        row_end = SIZE_IMG; col_end = SIZE_IMG;
        count = 0; % subfile count
        specStruct(j).fileName = fileName;
        specStruct(j).localIndices = localIndices;
        runLoopRow = true;
        while (row_end <= numRows)
            runLoopCol = true;
            while(col_end <= numCols)
                count = count + 1;
                fprintf('dir = %d, image = %d, subimage = %d\n', i, j, count);
                
                rmin = row_st + rowOffset; %+ localIndices.rowOffset;
                if localIndices.rsize < SIZE_IMG
                    rmax = rmin + localIndices.rsize - 1;
                else
                    rmax = rmin + SIZE_IMG - 1;
                end
                
                cmin = col_st + colOffset; % + localIndices.colOffset;
                if localIndices.csize < SIZE_IMG
                    cmax = cmin + localIndices.csize - 1;
                else
                    cmax = cmin + SIZE_IMG - 1;
                end                
                specStruct(j).globalIndices = [specStruct(j).globalIndices; ...
                    [rmin, rmax, cmin, cmax]];
                subfileName = [fileName(1:end-7), ...
                        num2str(count), '_', fileName(end-6:end)];
                specStruct(j).subfiles{count} = subfileName;
                imwrite(im(row_st : row_end, col_st : col_end, :), ...
                        fullfile(tmpOutRgbPath, subfileName));
                col_st = col_st + SIZE_STEP;
                col_end = col_end + SIZE_STEP;                
%                if j == 6 % debug
%                    disp(j);
%                end
                if (col_end > numCols) && (col_end ~= numCols + SIZE_STEP) ...
                                    && (runLoopCol == true)
                    col_end = numCols;
                    col_st = numCols - SIZE_IMG + 1;
                	runLoopCol = false;
                end
            end
            row_st = row_st + SIZE_STEP;            
            row_end = row_end + SIZE_STEP;
            col_st = 1;
            col_end = SIZE_IMG;
            if (row_end > numRows) && (row_end ~= numRows + SIZE_STEP) ...
                                    && (runLoopRow == true)
                row_end = numRows;
                row_st = numRows - SIZE_IMG + 1;
                runLoopRow = false;
            end            
        end
    end
    
    switch i
        case 1
            train_binSeg_A1 = specStruct;
        case 2
            train_binSeg_A2 = specStruct;
        case 3
            train_binSeg_A3 = specStruct;
        case 4
            train_binSeg_A4 = specStruct;            
        otherwise
            error('Index out of range');
    end
    clear specStruct;
end

% save all mat files
save('./train_binSeg.mat', 'train_binSeg_A1', 'train_binSeg_A2', ...
        'train_binSeg_A3', 'train_binSeg_A4');
restoredefaultpath;

end

% % function box = getBboxIndices(bb, bb_f, bb_flr, bb_fflr, numRows, numCols)
% % 
% %     bb = round(bb);
% %     bb_f = round(bb_f);
% %     bb_flr = round(bb_flr);
% %     bb_fflr = round(bb_fflr);
% %     
% %     box = struct();
% %     
% %     % original and its bottom-up rotated version
% %     c_min1 = min(bb(:,1));
% %     c_max1 = max(bb(:,1) + bb(:,3) - 1);
% %     r_min1 = min(bb(:,2));
% %     r_max1 = max(bb(:,2) + bb(:,4) - 1);   
% %     
% %     c_min1_f = min(bb_f(:,1));
% %     c_max1_f = max(bb_f(:,1) + bb_f(:,3) - 1);
% %     r_min1_f = min(bb_f(:,2));
% %     r_max1_f = max(bb_f(:,2) + bb_f(:,4) - 1);
% %     
% %     r_min_01 = max(r_min1, numRows - r_max1_f + 1);
% %     r_max_01 = min(r_max1, numRows - r_min1_f + 1);
% %     c_min_01 = max(c_min1, c_min1_f);
% %     c_max_01 = min(c_max1, c_max1_f);
% %     
% %     % fliplr and its bottom-up rotated version
% %     c_min2 = min(bb_flr(:,1));
% %     c_max2 = max(bb_flr(:,1) + bb_flr(:,3) - 1);
% %     r_min2 = min(bb_flr(:,2));
% %     r_max2 = max(bb_flr(:,2) + bb_flr(:,4) - 1);   
% %     
% %     c_min2_f = min(bb_fflr(:,1));
% %     c_max2_f = max(bb_fflr(:,1) + bb_fflr(:,3) - 1);
% %     r_min2_f = min(bb_fflr(:,2));
% %     r_max2_f = max(bb_fflr(:,2) + bb_fflr(:,4) - 1);
% %     
% %     r_min_02 = max(r_min2, numRows - r_max2_f + 1);
% %     r_max_02 = min(r_max2, numRows - r_min2_f + 1);
% %     c_min_02 = min(c_max2, c_max2_f);
% %     c_max_02 = max(c_min2, c_min2_f);
% %     c_min_02 = numCols - c_min_02 + 1;
% %     c_max_02 = numCols - c_max_02 + 1;
% %     
% %     box.rmin = max(r_min_01, r_min_02);
% %     box.rmax = min(r_max_01, r_max_02);
% %     box.cmin = max(c_min_01, c_min_02);
% %     box.cmax = min(c_max_01, c_max_02);
% % %    box.r_min2 = box.r_min1;
% % %    box.r_max2 = box.r_max1;
% % %    box.c_min2 = numCols - box.c_max1 + 1;
% % %    box.c_max2 = numCols - box.c_min1 + 1;   
% %     
% % end

function [im_out, localIndices] = correctImgSize(im, size_)
    [r,c,~] = size(im);
    
    if (r >= size_) && (c >= size_)
        im_out = im;
        localIndices = struct('rmin', 1, 'rmax', r, 'cmin', 1, ...
            'cmax', c, 'rowOffset', 0, 'colOffset', 0, ...
            'rsize', r, 'csize', c);
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
    
    im_out = uint8(zeros(r_size, c_size, 3));
    im_out(r_lb:r_ub, c_lb:c_ub, :) = im;
    
    localIndices = struct('rmin', r_lb, 'rmax', r_ub, 'cmin', c_lb, ...
        'cmax', c_ub, 'rowOffset', r_lb-1, 'colOffset', c_lb-1, ...
        'rsize', r_ub-r_lb+1, 'csize', c_ub-c_lb+1);
end
