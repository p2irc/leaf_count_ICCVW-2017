function genFineTuneBinSegData_nobox()

restoredefaultpath;

% -------------------- system input ---------------------- %
rng(72);
global g_SIZE_IMG g_SIZE_STEP g_NumOutDirs;
g_SIZE_IMG = 224;
g_SIZE_STEP = 100; % step size to generate image from the original
g_NumOutDirs = 50; % number of directories for the images
inBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/lcc_original';
outBasePath = '/media/aich/DATA/databases/lcc_finetune_nobox';
rgbPath = 'rgb';
fgPath = 'fg';

inRgbPath = fullfile(inBasePath, rgbPath);
inFgPath = fullfile(inBasePath, fgPath);

global g_OutRgbPath g_OutFgPath;
g_OutRgbPath = fullfile(outBasePath, rgbPath);
g_OutFgPath = fullfile(outBasePath, fgPath);

assert(isdir(inRgbPath), 'Input RGB directory does not exist\n %s', inRgbPath);
assert(isdir(inFgPath), 'Input FG directory does not exist\n %s', inFgPath);

% check if directory exists, remove old directory and create new ones
if isdir(outBasePath)
    assert(rmdir(outBasePath, 's'), 'Cannot remove old base directory\n %s', outBasePath);
end
if isdir(g_OutRgbPath)
    assert(rmdir(g_OutRgbPath, 's'), 'Cannot remove old RGB directory\n %s', g_OutRgbPath);
end
if isdir(g_OutFgPath)
    assert(rmdir(g_OutFgPath, 's'), 'Cannot remove old FG directory\n %s', g_OutFgPath);
end
assert(mkdir(outBasePath), 'Cannot create new base directory\n %s', outBasePath);
assert(mkdir(g_OutRgbPath), 'Cannot create new RGB directory\n %s', g_OutRgbPath);
assert(mkdir(g_OutFgPath), 'Cannot create new FG directory\n %s', g_OutFgPath);

for i = 1:g_NumOutDirs
    tmpRgbPath = fullfile(g_OutRgbPath, num2str(i));
    tmpFgPath = fullfile(g_OutFgPath, num2str(i));
    assert(mkdir(tmpRgbPath), 'Cannot create new RGB directory\n %s', tmpRgbPath);
    assert(mkdir(tmpFgPath), 'Cannot create new FG directory\n %s', tmpFgPath);
end

global g_NumFilesPerDir;
g_NumFilesPerDir = single(zeros(g_NumOutDirs, 1));
%------------------------------------------------------------------%

imgList = dir(fullfile(inRgbPath, '*.png'));
for i = 1:length(imgList) 
    fprintf('file = %d\n', i);
    im = imread(fullfile(inRgbPath, imgList(i).name));
    gtb = imread(fullfile(inFgPath, imgList(i).name)) > 0;
    getSaveImages(im, gtb);
    getSaveImages(flip(im), flip(gtb));
    getSaveImages(fliplr(im), fliplr(gtb));
    getSaveImages(flip(fliplr(im)), flip(fliplr(gtb))); 
end

end

function getSaveImages(im, gtb)
    
    global g_im_ g_gtb_;
    global g_SIZE_IMG g_SIZE_STEP;
    size_ = g_SIZE_IMG;
    step_ = g_SIZE_STEP;
    [numRows, numCols, ~] = size(im);
    rlim = numRows - size_ + 1;
    clim = numCols - size_ + 1;
    for i = 1:step_:rlim
        for j = 1:step_:clim
            g_im_ = im(i:i+size_-1, j:j+size_-1,:);
            g_gtb_ = gtb(i:i+size_-1, j:j+size_-1);
            saveSingleImage();
        end
        if j < clim
            j = clim;
            g_im_ = im(i:i+size_-1, j:j+size_-1,:);
            g_gtb_ = gtb(i:i+size_-1, j:j+size_-1);
            saveSingleImage();
        end
    end
    if i < rlim
        i = rlim;
        g_im_ = im(i:i+size_-1, j:j+size_-1,:);
        g_gtb_ = gtb(i:i+size_-1, j:j+size_-1);
        saveSingleImage();
    end
    
end

function saveSingleImage()
    global g_im_ g_gtb_;
    global g_NumFilesPerDir g_NumOutDirs g_OutRgbPath g_OutFgPath;
    dirInd = randi(g_NumOutDirs);
    g_NumFilesPerDir(dirInd) = g_NumFilesPerDir(dirInd) + 1;
    tmpFileName = [num2str(g_NumFilesPerDir(dirInd)), '.png'];
    tmpOutRgbPath = fullfile(g_OutRgbPath, num2str(dirInd), tmpFileName);
    tmpOutFgPath = fullfile(g_OutFgPath, num2str(dirInd), tmpFileName);
    imwrite(g_im_, tmpOutRgbPath);
    imwrite(g_gtb_, tmpOutFgPath);
end


