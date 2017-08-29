clear all; close all; %clc;

dataPath = '/media/aich/DATA/databases/leaf_cvppp2017/test_count/rgb_resize';
listFileName = 'testFileLists.mat';
countFileName = 'testFileCounts.mat';
outPath = 'AR/Aich_S_results/';
outBasePath = 'AR';
if isdir(outBasePath)
    rmdir(outBasePath, 's');
end
mkdir(outPath);

load(fullfile(dataPath, listFileName));
load(fullfile(dataPath, countFileName));

countList_a1 = uint8(round(countList_a1));
countList_a2 = uint8(round(countList_a2));
countList_a3 = uint8(round(countList_a3));
countList_a4 = uint8(round(countList_a4));
countList_a5 = uint8(round(countList_a5));

countList_a1(countList_a1==0) = 1;
countList_a2(countList_a2==0) = 1;
countList_a3(countList_a3==0) = 1;
countList_a4(countList_a4==0) = 1;
countList_a5(countList_a5==0) = 1;
fileList_a1 = fileList_a1';
fileList_a2 = fileList_a2';
fileList_a3 = fileList_a3';
fileList_a4 = fileList_a4';
fileList_a5 = fileList_a5';

writetable(table(fileList_a1, countList_a1), ...
        fullfile(outPath, 'A1.csv'), ...
        'WriteVariableNames',false);
    
writetable(table(fileList_a2, countList_a2), ...
        fullfile(outPath, 'A2.csv'), ...
        'WriteVariableNames',false);
    
writetable(table(fileList_a3, countList_a3), ...
        fullfile(outPath, 'A3.csv'), ...
        'WriteVariableNames',false);
    
writetable(table(fileList_a4, countList_a4), ...
        fullfile(outPath, 'A4.csv'), ...
        'WriteVariableNames',false);
    
writetable(table(fileList_a5, countList_a5), ...
        fullfile(outPath, 'A5.csv'), ...
        'WriteVariableNames',false);    