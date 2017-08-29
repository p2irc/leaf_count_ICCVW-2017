% this file separates/generates test RGB and binary Fg files into 
% separate directories maintaining the subdirectory structure, e.g.
% A1, A2, A3, A4. This is the first file to be run for evaluating
% the binary segmentation model. 
% Execute genTestDataBinSeg.m after this file.


clear all; close all; clc;

baseImPath = '/media/aich/DATA/databases/leaf_cvppp2017/CVPPP2017_testing/testing/';
postImPath = {'A1','A2','A3','A4','A5'};
outBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/test_binSeg';
outRgbPath = 'rgb';
outFgPath = 'fg';

outRgbPath = fullfile(outBasePath, outRgbPath);
outFgPath = fullfile(outBasePath, outFgPath);

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

% create sub directories
for i = 1:length(postImPath)
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    tmpOutFgPath = fullfile(outFgPath, postImPath{i});    
    if isdir(tmpOutRgbPath)
        assert(rmdir(tmpOutRgbPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutRgbPath);
    end
    if isdir(tmpOutFgPath)
        assert(rmdir(tmpOutFgPath, 's'), ...
            'Cannot remove old FG directory\n %s', tmpOutFgPath);
    end    
    assert(mkdir(tmpOutRgbPath), ...
        'Cannot create RGB subdirectory\n %s', tmpOutRgbPath);
    assert(mkdir(tmpOutFgPath), ...
        'Cannot create FG subdirectory\n %s', tmpOutFgPath);
end
% ----------------------------------------------------------------------


count = 0; % count files
for i = 1:length(postImPath)
    tmpImPath = fullfile(baseImPath, postImPath{i});
    tmpOutRgbPath = fullfile(outRgbPath, postImPath{i});
    tmpOutFgPath = fullfile(outFgPath, postImPath{i});
    
    rgbPaths = dir(fullfile(tmpImPath, '*_rgb.png'));
    for j = 1:length(rgbPaths)
        count = count + 1;
        fprintf('file = %d\n', count);
        % copy file into rgb directory
        assert(copyfile(fullfile(tmpImPath, rgbPaths(j).name), tmpOutRgbPath), ...
            'Cannot copy RGB file = %s', fullfile(tmpImPath, rgbPaths(j).name));

        fgFileName = [rgbPaths(j).name(1:end-7), 'fg.png'];
        % copy file into fg directory
        assert(copyfile(fullfile(tmpImPath, fgFileName), tmpOutFgPath), ...
            'Cannot copy FG file = %s', fullfile(tmpImPath, fgFileName));
    end
end