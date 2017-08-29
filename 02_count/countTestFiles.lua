require 'torch';
require 'paths';
require 'image';
local matio = require 'matio';
matio.use_lua_strings = true; -- to read file names as strings than char tensors

torch.setdefaulttensortype('torch.FloatTensor');

---------- model loading start ---------
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

cudnn.benchmark = true; -- true does not work variable size

--model = torch.load('./models/model_count_v1_epoch_83.t7'); 
model = torch.load('./models/model_count_reducer_02_epoch_35.t7'); 
model:evaluate();
--model:training();
model:cuda();
---------- model loading finished ---------

--local mean_seg_finetune = torch.load('mean_seg_finetune.dat'); -- image mean
local dataPath = '/media/aich/DATA/databases/leaf_cvppp2017/test_count/rgb_resize';
local bsPath = '/media/aich/DATA/databases/leaf_cvppp2017/test_count/bs_resize';
local postPath = {[1]='A1', [2]='A2', [3]='A3', [4]='A4', [5]='A5'};
local listFileName = 'testFileLists.mat';
local countFileName = 'testFileCounts.mat';
local fileList;

function runTest()
	-- load one directory list at a time 'A1','A2','A3','A4','A5'

	-- directory 'A1'
	fileList = matio.load(paths.concat(dataPath, listFileName), 'fileList_a1');
	local countList_a1 = testSingleFile(paths.concat(dataPath, postPath[1]), paths.concat(bsPath, postPath[1]));
	-- directory 'A2'	
	fileList = matio.load(paths.concat(dataPath, listFileName), 'fileList_a2');
	local countList_a2 = testSingleFile(paths.concat(dataPath, postPath[2]), paths.concat(bsPath, postPath[2]));
	-- directory 'A3'	
	fileList = matio.load(paths.concat(dataPath, listFileName), 'fileList_a3');
	local countList_a3 = testSingleFile(paths.concat(dataPath, postPath[3]), paths.concat(bsPath, postPath[3]));
	-- directory 'A4'
	fileList = matio.load(paths.concat(dataPath, listFileName), 'fileList_a4');
	local countList_a4 = testSingleFile(paths.concat(dataPath, postPath[4]), paths.concat(bsPath, postPath[4]));
	-- directory 'A5'	
	fileList = matio.load(paths.concat(dataPath, listFileName), 'fileList_a5');
	local countList_a5 = testSingleFile(paths.concat(dataPath, postPath[5]), paths.concat(bsPath, postPath[5]));

	-- save count files
	matio.save(paths.concat(dataPath, countFileName), {countList_a1=countList_a1, 
					countList_a2=countList_a2, countList_a3=countList_a3,
					countList_a4=countList_a4,countList_a5=countList_a5});

end

local inputsCpu = torch.Tensor();
--local inputs = torch.CudaTensor();

function testSingleFile(rgbPath, bsPath)
	print('====== Entering new directory ======');
	local countList = torch.Tensor(#fileList):fill(0.0);
	for i = 1, #fileList, 1 do
--		print(tostring(fileList[i]));
		local im = image.load(paths.concat(rgbPath, fileList[i]), 3, 'float'); -- - mean_seg_finetune;
		local gt = image.load(paths.concat(bsPath, fileList[i]), 1, 'float');
		local inputsCpu = torch.cat(im, gt, 1);
		local outputs = model:forward(inputsCpu:cuda());
--		print(torch.type(outputs));
--		print(outputs:size());
		countList[i] = outputs:float()[1];
		print(tostring(fileList[i]), tostring(outputs:float()[1]));
	end

	return countList;
end





