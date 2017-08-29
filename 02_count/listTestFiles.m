function listTestFiles()

dataPath = '/media/aich/DATA/databases/leaf_cvppp2017/test_count/rgb_resize';
postPath = {'A1','A2','A3','A4','A5'};

imgList = dir(fullfile(dataPath, postPath{1}, '*.png'));
fileList_a1 = getNames(imgList);
imgList = dir(fullfile(dataPath, postPath{2}, '*.png'));
fileList_a2 = getNames(imgList);
imgList = dir(fullfile(dataPath, postPath{3}, '*.png'));
fileList_a3 = getNames(imgList);
imgList = dir(fullfile(dataPath, postPath{4}, '*.png'));
fileList_a4 = getNames(imgList);
imgList = dir(fullfile(dataPath, postPath{5}, '*.png'));
fileList_a5 = getNames(imgList);


save(fullfile(dataPath, 'testFileLists.mat'), 'fileList_a1', ...
    'fileList_a2', 'fileList_a3', 'fileList_a4', 'fileList_a5');

end

function fileList = getNames(imgList)

fileList = {};
for i=1:length(imgList)
    fileList{length(fileList)+1} = imgList(i).name;
end

end