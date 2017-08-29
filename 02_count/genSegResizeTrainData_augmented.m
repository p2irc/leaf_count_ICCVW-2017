function genSegResizeTrainData_augmented()

restoredefaultpath;

% -------------------- system input ---------------------- %
%global g_NumDirs;
%g_NumDirs = 5; % number of sub-directories where images are stored
global g_OutDataPath g_OutBsPath;

basePath = '/media/aich/DATA/databases/leaf_cvppp2017/train_count';
inCsvPath = basePath;
inRgbPath = 'rgb_resize';
inBinaryPath = 'bs_resize';
postImPath = {'A1','A2','A3','A4'};

g_OutDataPath = 'data_seg_aug';
g_OutBsPath = 'bs_seg_aug';
gtFileName = 'gt_seg_aug.mat';

global g_GaussFilt_1 g_GaussFilt_2;
g_GaussFilt_1 = fspecial('gaussian', 5, 1);
g_GaussFilt_2 = fspecial('gaussian', 9, 2);
% --------------------------------------------------------- %

g_OutDataPath = fullfile(basePath, g_OutDataPath);
g_OutBsPath = fullfile(basePath, g_OutBsPath);
inRgbPath = fullfile(basePath, inRgbPath);
inBinaryPath = fullfile(basePath, inBinaryPath);
gtFileName = fullfile(basePath, gtFileName);

% remove old augmented directory with subdirectories
if isdir(g_OutDataPath)
    assert(rmdir(g_OutDataPath, 's'), ...
        'Cannot remove old data directory\n %s', g_OutDataPath);
end
assert(mkdir(g_OutDataPath), 'Cannot create new data directory\n %s', g_OutDataPath);
if isdir(g_OutBsPath)
    assert(rmdir(g_OutBsPath, 's'), ...
        'Cannot remove old data directory\n %s', g_OutBsPath);
end
assert(mkdir(g_OutBsPath), 'Cannot create new data directory\n %s', g_OutBsPath);


% % create new augmented subdirectories
% for i = 1:g_NumDirs
%     assert(mkdir(fullfile(outDataPath, num2str(i))), ...
%         'Cannot create new data subdirectory\n %s', ...
%         fullfile(outDataPath, num2str(i)));
% end
% % ----------------------------------------------------------------------

global g_im_rgb g_im_bin;
global g_Count g_NumLeaves g_GtAug;
g_Count = 0;
g_GtAug = [];
%global g_NumImgPerDir;
%g_NumImgPerDir = single(zeros(numDirs, 1));
for i = 1:length(postImPath) % subdirectory index
    tmpInRgbPath = fullfile(inRgbPath, postImPath{i});
    tmpInBinaryPath = fullfile(inBinaryPath, postImPath{i});
    tmpInCsvPath = fullfile(inCsvPath, [postImPath{i}, '.csv']);
    labelsCsv = readtable(tmpInCsvPath);

    for j = 1:size(labelsCsv,1) % file inside subdir index
        fprintf('subdir = %d, file = %d\n', i, j);
        imgName = labelsCsv{j,1}{1};
        g_NumLeaves = labelsCsv{j,2}; % #leaves
        g_im_rgb = imread(fullfile(tmpInRgbPath, imgName));
        g_im_bin = imread(fullfile(tmpInBinaryPath, imgName));
        modBgIntensity();   
    end
end

assert(length(g_GtAug) == g_Count, 'Gt and Data dimensions mismatch');
save(gtFileName, 'g_GtAug');

end

function modBgIntensity()
    global g_im_rgb g_im_bin g_im_rgb_mod g_im_bin_mod;
    global g_GaussFilt_1 g_GaussFilt_2;

    g_im_rgb_mod = g_im_rgb;
    g_im_bin_mod = g_im_bin;
    %figure; imshow(g_im_rgb);
    saveImages();
    % ------------------------------------------------------- %
    % adjust contrast
    g_im_rgb_mod = cat(3, imadjust(g_im_rgb(:,:,1)), ...
        imadjust(g_im_rgb(:,:,2)), imadjust(g_im_rgb(:,:,3)));
    tmp1 = g_im_rgb_mod(:,:,1);
    tmp2 = g_im_rgb(:,:,1);    
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,1) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,2);
    tmp2 = g_im_rgb(:,:,2);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,2) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,3);
    tmp2 = g_im_rgb(:,:,3);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,3) = tmp1;
    %figure; imshow(g_im_rgb_mod);
    saveImages();
% % %     % ------------------------------------------------------- %
% % %     % histogram equalization
% % %     tileSize1 = floor(size(g_im_bin,1)/20);
% % %     tileSize2 = floor(size(g_im_bin,2)/20);
% % %     g_im_rgb_mod = cat(3, ...
% % %         adapthisteq(g_im_rgb(:,:,1), 'NumTiles', [tileSize1, tileSize2]), ...
% % %         adapthisteq(g_im_rgb(:,:,2), 'NumTiles', [tileSize1, tileSize2]), ...
% % %         adapthisteq(g_im_rgb(:,:,3), 'NumTiles', [tileSize1, tileSize2]) );
% % %     tmp1 = g_im_rgb_mod(:,:,1);
% % %     tmp2 = g_im_rgb(:,:,1);    
% % %     tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
% % %     g_im_rgb_mod(:,:,1) = tmp1;
% % %     tmp1 = g_im_rgb_mod(:,:,2);
% % %     tmp2 = g_im_rgb(:,:,2);
% % %     tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
% % %     g_im_rgb_mod(:,:,2) = tmp1;
% % %     tmp1 = g_im_rgb_mod(:,:,3);
% % %     tmp2 = g_im_rgb(:,:,3);
% % %     tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
% % %     g_im_rgb_mod(:,:,3) = tmp1;
% % %     %figure; imshow(g_im_rgb_mod); 
% % %     saveImages();
    % ------------------------------------------------------- %
    % blur 1
    g_im_rgb_mod = imfilter(g_im_rgb, g_GaussFilt_1, 'same', 'conv');
    %figure; imshow(g_im_rgb_mod);
    saveImages();
    % ------------------------------------------------------- %
    % blur 1 - original 
    tmp1 = g_im_rgb_mod(:,:,1);
    tmp2 = g_im_rgb(:,:,1);    
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,1) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,2);
    tmp2 = g_im_rgb(:,:,2);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,2) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,3);
    tmp2 = g_im_rgb(:,:,3);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,3) = tmp1;
    %figure; imshow(g_im_rgb_mod);     
    saveImages();
    % ------------------------------------------------------- %
    % blur 2
    g_im_rgb_mod = imfilter(g_im_rgb, g_GaussFilt_2, 'same', 'conv');
    %figure; imshow(g_im_rgb_mod);
    saveImages();
    % ------------------------------------------------------- %
    % blur 2 - original 
    tmp1 = g_im_rgb_mod(:,:,1);
    tmp2 = g_im_rgb(:,:,1);    
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,1) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,2);
    tmp2 = g_im_rgb(:,:,2);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,2) = tmp1;
    tmp1 = g_im_rgb_mod(:,:,3);
    tmp2 = g_im_rgb(:,:,3);
    tmp1(g_im_bin==1) = tmp2(g_im_bin==1);
    g_im_rgb_mod(:,:,3) = tmp1;
    %figure; imshow(g_im_rgb_mod);     
    saveImages();    
    % ------------------------------------------------------- %
    % sharp 1    
    g_im_rgb_mod = imsharpen(g_im_rgb, 'Radius', 3, 'Amount', 1);
    %figure; imshow(g_im_rgb_mod);
    saveImages();
    % ------------------------------------------------------- %
    % sharp 2    
    g_im_rgb_mod = imsharpen(g_im_rgb, 'Radius', 5, 'Amount', 1);
    %figure; imshow(g_im_rgb_mod);
    saveImages();
    % ------------------------------------------------------- %
    % noise 1
    g_im_rgb_mod = imnoise(g_im_rgb, 'gaussian', 0, 0.001);
    %figure; imshow(g_im_rgb_mod);
    saveImages();
    % ------------------------------------------------------- %
    
end

function saveImages()
    global g_im_rgb_mod g_im_bin_mod;
    global g_OutDataPath g_OutBsPath;
    global g_Count g_NumLeaves g_GtAug;

    im_easy = g_im_rgb_mod;
    bs_easy = g_im_bin_mod;
    g_Count = g_Count + 1;  
    g_GtAug = [g_GtAug; g_NumLeaves];
    imwrite(im_easy, fullfile(g_OutDataPath, [num2str(g_Count), '.png']));
    imwrite(bs_easy>0, fullfile(g_OutBsPath, [num2str(g_Count), '.png']));
    im_tmp = im_easy; % store main version
    bs_tmp = bs_easy;
    im_easy = flip(im_easy); % top-bottom flip
    bs_easy = flip(bs_easy);
    g_Count = g_Count + 1;    
    g_GtAug = [g_GtAug; g_NumLeaves];
    imwrite(im_easy, fullfile(g_OutDataPath, [num2str(g_Count), '.png']));
    imwrite(bs_easy>0, fullfile(g_OutBsPath, [num2str(g_Count), '.png']));
    im_easy = fliplr(im_easy); % 180 rotation
    bs_easy = fliplr(bs_easy);
    g_Count = g_Count + 1;    
    g_GtAug = [g_GtAug; g_NumLeaves];
    imwrite(im_easy, fullfile(g_OutDataPath, [num2str(g_Count), '.png']));
    imwrite(bs_easy>0, fullfile(g_OutBsPath, [num2str(g_Count), '.png']));
    im_easy = fliplr(im_tmp); % lr-flip
    bs_easy = fliplr(bs_tmp);
    g_Count = g_Count + 1;    
    g_GtAug = [g_GtAug; g_NumLeaves];
    imwrite(im_easy, fullfile(g_OutDataPath, [num2str(g_Count), '.png']));
    imwrite(bs_easy>0, fullfile(g_OutBsPath, [num2str(g_Count), '.png']));
end
