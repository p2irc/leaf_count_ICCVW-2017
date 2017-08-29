clear all; close all; clc;

inBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/test_binSeg';
inRgbPath = 'rgb';
inFgPath = 'fg';
inBsPath = 'bs_sum_plain_nobox';
postImPath = {'A1','A2','A3','A4','A5'};
outBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/test_count';
outRgbPath = 'rgb';
outFgPath = 'fg';
outBsPath = 'bs';


inRgbPath = fullfile(inBasePath, inRgbPath);
inFgPath = fullfile(inBasePath, inFgPath);
inBsPath = fullfile(inBasePath, inBsPath);
outRgbPath = fullfile(outBasePath, outRgbPath);
outFgPath = fullfile(outBasePath, outFgPath);
outBsPath = fullfile(outBasePath, outBsPath);

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
if isdir(outBsPath)
    assert(rmdir(outFgPath, 's'), 'Cannot remove old BS directory\n %s', outFgPath);
end
assert(mkdir(outBasePath), 'Cannot create new base directory\n %s', outBasePath);
assert(mkdir(outRgbPath), 'Cannot create new RGB directory\n %s', outRgbPath);
assert(mkdir(outFgPath), 'Cannot create new FG directory\n %s', outFgPath);
assert(mkdir(outBsPath), 'Cannot create new BS directory\n %s', outBsPath);

% create sub directories
for i = 1:length(postImPath)
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    tmpOutFgPath = fullfile(outFgPath, postImPath{i});    
    tmpOutBsPath = fullfile(outBsPath, postImPath{i});        
    if isdir(tmpOutRgbPath)
        assert(rmdir(tmpOutRgbPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutRgbPath);
    end
    if isdir(tmpOutFgPath)
        assert(rmdir(tmpOutFgPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutFgPath);
    end    
    if isdir(tmpOutBsPath)
        assert(rmdir(tmpOutBsPath, 's'), ...
            'Cannot remove old BS directory\n %s', tmpOutBsPath);
    end        
    assert(mkdir(tmpOutRgbPath), ...
        'Cannot create RGB subdirectory\n %s', tmpOutRgbPath);
    assert(mkdir(tmpOutFgPath), ...
        'Cannot create FG subdirectory\n %s', tmpOutFgPath);
    assert(mkdir(tmpOutBsPath), ...
        'Cannot create BS subdirectory\n %s', tmpOutBsPath);    
end
% ----------------------------------------------------------------------


count = 0; % count files
for i = 1:length(postImPath)
    tmpInRgbPath = fullfile(inRgbPath, postImPath{i});
    tmpInFgPath = fullfile(inFgPath, postImPath{i});
    tmpInBsPath = fullfile(inBsPath, postImPath{i});
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    tmpOutFgPath = fullfile(outFgPath, postImPath{i});
    tmpOutBsPath = fullfile(outBsPath, postImPath{i});
    rgbPaths = dir(fullfile(tmpInRgbPath, '*_rgb.png'));    
    for j = 1:length(rgbPaths)
        count = count + 1;
        fprintf('file = %d\n', count);
        % copy file into rgb directory
        assert(copyfile(fullfile(tmpInRgbPath, rgbPaths(j).name), tmpOutRgbPath), ...
            'Cannot copy RGB file = %s', fullfile(tmpInRgbPath, rgbPaths(j).name));

        fgFileName = [rgbPaths(j).name(1:end-7), 'fg.png'];
        % copy file into fg directory
        assert(copyfile(fullfile(tmpInFgPath, fgFileName), tmpOutFgPath), ...
            'Cannot copy FG file = %s', fullfile(tmpInFgPath, fgFileName));
        
        bsFileName = rgbPaths(j).name;
        % copy file into fg directory
        assert(copyfile(fullfile(tmpInBsPath, bsFileName), tmpOutBsPath), ...
            'Cannot copy BS file = %s', fullfile(tmpInBsPath, bsFileName));        
    end
end
