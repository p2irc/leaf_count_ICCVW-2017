% create 2 versions of binary segmentation, 
% single_max : label with single maximum probability
% sum_max : label with summed up maximum probability

clear all; close all; clc;

SIZE_IMG = 224; 
load train_binSeg.mat;

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/';
inPath = fullfile(basePath, 'fg_mat_nobox');
postPath = {'A1','A2','A3','A4'};
outPath_single = fullfile(basePath,'bs_single_nobox');
outPath_sum = fullfile(basePath, 'bs_sum_nobox');
outPath_single_plain = fullfile(basePath, 'bs_single_plain_nobox');
outPath_sum_plain = fullfile(basePath, 'bs_sum_plain_nobox');

listFailure = {}; % list of failed files
failFile = false; % boolean to check whether file is failed 
% create sub directories
for i = 1:length(postPath)
    tmpOutPath_single = fullfile(outPath_single, postPath{i});
    tmpOutPath_sum = fullfile(outPath_sum, postPath{i});
    tmpOutPath_single_plain = fullfile(outPath_single_plain, postPath{i});
    tmpOutPath_sum_plain = fullfile(outPath_sum_plain, postPath{i});
    
    if isdir(tmpOutPath_single)
        assert(rmdir(tmpOutPath_single, 's'), ...
            'Cannot remove old BS(single) directory\n %s', tmpOutPath_single);
    end
    assert(mkdir(tmpOutPath_single), ...
        'Cannot create BS(single) subdirectory\n %s', tmpOutPath_single);
    
    if isdir(tmpOutPath_sum)
        assert(rmdir(tmpOutPath_sum, 's'), ...
            'Cannot remove old BS(sum) directory\n %s', tmpOutPath_sum);
    end
    assert(mkdir(tmpOutPath_sum), ...
        'Cannot create BS(sum) subdirectory\n %s', tmpOutPath_sum);   
    
    if isdir(tmpOutPath_single_plain)
        assert(rmdir(tmpOutPath_single_plain, 's'), ...
            'Cannot remove old BS(single plain) directory\n %s', tmpOutPath_single_plain);
    end
    assert(mkdir(tmpOutPath_single_plain), ...
        'Cannot create BS(single plain) subdirectory\n %s', tmpOutPath_single_plain);
    
    if isdir(tmpOutPath_sum_plain)
        assert(rmdir(tmpOutPath_sum_plain, 's'), ...
            'Cannot remove old BS(sum plain) directory\n %s', tmpOutPath_sum_plain);
    end
    assert(mkdir(tmpOutPath_sum_plain), ...
        'Cannot create BS(sum plain) subdirectory\n %s', tmpOutPath_sum_plain);       
end

for i = 1:length(postPath)
    switch i
        case 1
            specStruct = train_binSeg_A1;
        case 2
            specStruct = train_binSeg_A2;
        case 3
            specStruct = train_binSeg_A3;
        case 4
            specStruct = train_binSeg_A4;
        otherwise
            error('Index out of range');
    end
    
    fprintf('dir = %s, #files = %d\n', postPath{i}, length(specStruct));
    tmpInPath = fullfile(inPath, postPath{i});
    tmpOutPath_sum = fullfile(outPath_sum, postPath{i});
    tmpOutPath_single = fullfile(outPath_single, postPath{i});
    tmpOutPath_sum_plain = fullfile(outPath_sum_plain, postPath{i});
    tmpOutPath_single_plain = fullfile(outPath_single_plain, postPath{i});    
    for j = 1:length(specStruct)
%        fprintf('dir = %d, file = %d\n', i, j);
        % specify bs file names with full path
% %         fileName_sum = [specStruct(j).fileName(1:end-7), 'bs_sum.png'];
% %         fileName_single = [specStruct(j).fileName(1:end-7), 'bs_single.png'];
% %         fileName_sum_plain = [specStruct(j).fileName(1:end-7), 'bs_sum_plain.png'];
% %         fileName_single_plain = [specStruct(j).fileName(1:end-7), 'bs_single_plain.png'];        
% %         fileName_sum = fullfile(tmpOutPath_sum, fileName_sum);
% %         fileName_single = fullfile(tmpOutPath_single, fileName_single);
% %         fileName_sum_plain = fullfile(tmpOutPath_sum_plain, fileName_sum_plain);
% %         fileName_single_plain = fullfile(tmpOutPath_single_plain, fileName_single_plain);        
        fileName_sum = fullfile(tmpOutPath_sum, specStruct(j).fileName);
        fileName_single = fullfile(tmpOutPath_single, specStruct(j).fileName);
        fileName_sum_plain = fullfile(tmpOutPath_sum_plain, specStruct(j).fileName);
        fileName_single_plain = fullfile(tmpOutPath_single_plain, specStruct(j).fileName);        
        % assign image sizes, both local and global
        rsize = specStruct(j).fileSize(1); % original image row size
        csize = specStruct(j).fileSize(2); % original image col size
        rsize_box = specStruct(j).localIndices.rsize; % bounding box rsize
        csize_box = specStruct(j).localIndices.csize; % bounding box csize
        % initialize single and sum max probability matrices
        prob_single_max = single(zeros(2, rsize, csize));
        prob_sum_max = single(zeros(2, rsize, csize));
        if rsize_box < SIZE_IMG
            rmin_box = specStruct(j).localIndices.rmin;
            rmax_box = specStruct(j).localIndices.rmax;
        else
            rmin_box = 1;
            rmax_box = SIZE_IMG;
        end
        if csize_box < SIZE_IMG
            cmin_box = specStruct(j).localIndices.cmin;
            cmax_box = specStruct(j).localIndices.cmax;
        else
            cmin_box = 1;
            cmax_box = SIZE_IMG;
        end        
        
        for sf = 1:length(specStruct(j).subfiles)
            fprintf('dir = %d, file = %d, subfile = %d\n', i, j, sf);
            sfName = fullfile(tmpInPath, ...
                        [specStruct(j).subfiles{sf}(1:end-4), '.mat']);
            % check localIndices rsize and csize, if any of them or both
            % are smaller than macro size, use localIndices rmin, rmax,
            % cmin, cmax to copy values, otherwise copy as a whole
            load(sfName);
%            imshow(label>0);
            % copy index in the original image
            rmin_org = specStruct(j).globalIndices(sf, 1);
            rmax_org = specStruct(j).globalIndices(sf, 2);
            cmin_org = specStruct(j).globalIndices(sf, 3);
            cmax_org = specStruct(j).globalIndices(sf, 4);
            prob_sum_max(:, rmin_org:rmax_org, cmin_org:cmax_org) = ...
                bsxfun(@plus, prob_sum_max(:, rmin_org:rmax_org, ...
                cmin_org:cmax_org), prob(:, rmin_box:rmax_box, ...
                cmin_box:cmax_box));
            tmp1 = prob_single_max(:, rmin_org:rmax_org, cmin_org:cmax_org);
            tmp2 = prob(:, rmin_box:rmax_box, cmin_box:cmax_box);
            tmp1(tmp1 < tmp2) = tmp2(tmp1 < tmp2);
            prob_single_max(:, rmin_org:rmax_org, cmin_org:cmax_org) = tmp1;
        end
        % postprocess probability matrices to generate binary segmentations
%        if j == 6 % debug
%            disp(j);
%        end
        [a, b] = max(prob_sum_max, [], 1);
        a = squeeze(a);
        c = a(b==2);
        t = (max(c)-min(c)) * 0.1;
        if isempty(t)
            failFile = true;
            t = 0;
        end
        bs_sum_max = squeeze(prob_sum_max(2,:,:)) > t;
        % plain max_sum
        bs_sum_max_plain = squeeze(b);
        bs_sum_max_plain = bsxfun(@minus, bs_sum_max_plain, 1);

        
        [a, b] = max(prob_single_max, [], 1);
        a = squeeze(a);
        c = a(b==2);
        t = otsuthresh(hist(c));
        %t = (max(c)-min(c)) * t;
        t = max(c)*t;
        if isempty(t)
            failFile = true;
            t = 0;
        end
        bs_single_max = squeeze(prob_single_max(2,:,:)) > t;
        bs_single_max_plain = squeeze(b);
        bs_single_max_plain = bsxfun(@minus, bs_single_max_plain, 1);        
%        figure(1); 
%        subplot(1,2,1); imshow(bs_sum_max>0);
%        subplot(1,2,2); imshow(bs_single_max>0);
        
        % save images
        imwrite(bs_sum_max, fileName_sum);
        imwrite(bs_single_max, fileName_single);
        imwrite(bs_sum_max_plain, fileName_sum_plain);
        imwrite(bs_single_max_plain, fileName_single_plain);        
        
        if failFile 
            listFailure{length(listFailure)+1} = ...
                fullfile(postPath{i}, specStruct(j).fileName);
            failFile = false;
        end
        
    end
end

save('Failed_Files_BS_Train_nobox.mat', 'listFailure');
