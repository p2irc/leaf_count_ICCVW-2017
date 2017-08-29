clear all; close all; clc;

baseImPath = '/media/aich/DATA/databases/leaf_cvppp2017/CVPPP2017_LSC_training/training/';
postImPath = {'A1','A2','A3','A4'};
outBasePath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_original';
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
%------------------------------------------------------------------%

count = 0; % count files
for i = 1:length(postImPath)
    tmpImPath = fullfile(baseImPath, postImPath{i});
    rgbPaths = dir(fullfile(tmpImPath, '*_rgb.png'));
    for j = 1:length(rgbPaths)
        count = count + 1;
        fprintf('file = %d\n', count);
        % copy file into rgb directory
        assert(copyfile(fullfile(tmpImPath, rgbPaths(j).name), outRgbPath), ...
            'Cannot copy RGB file = %s', fullfile(tmpImPath, rgbPaths(j).name));
        % rename file inside rgb directory
        assert(movefile(fullfile(outRgbPath, rgbPaths(j).name), ...
            fullfile(outRgbPath, [num2str(count), '.png'])), ...
            'Cannot rename RGB file = %s', fullfile(outRgbPath, rgbPaths(j).name));
        
        fgFileName = [rgbPaths(j).name(1:end-7), 'fg.png'];
        % copy file into fg directory
        assert(copyfile(fullfile(tmpImPath, fgFileName), outFgPath), ...
            'Cannot copy FG file = %s', fullfile(tmpImPath, fgFileName));
        % rename file inside fg directory
        assert(movefile(fullfile(outFgPath, fgFileName), ...
            fullfile(outFgPath, [num2str(count), '.png'])), ...
            'Cannot rename FG file = %s', fullfile(outFgPath, fgFileName));        
    end
end
