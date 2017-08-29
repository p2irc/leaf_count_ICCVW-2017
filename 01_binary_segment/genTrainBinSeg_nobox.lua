require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

require 'paths';
require 'image'; -- read test images
local matio = require 'matio'; -- deal with mat files

cudnn.benchmark = true;

torch.setdefaulttensortype('torch.FloatTensor');

local IMG_MAX = 255;

local pathInTestData = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/rgb_224_nobox/';
local pathOutTestData = '/media/aich/DATA/databases/leaf_cvppp2017/train_binSeg/fg_mat_nobox/';
local subDirs = {[1]='A1', [2]='A2', [3]='A3', [4]='A4'};
local meanData = torch.load("mean_finetune_nobox_3chn.dat")/IMG_MAX;

local batchSize, inSize, inDim = 1, 224, 3;

local inputsCpu = torch.Tensor(batchSize, inDim, inSize, inSize);
local inputs = torch.CudaTensor(batchSize, inDim, inSize, inSize); -- float

-- load model
local model = torch.load('./models/model_segnet_v1_epoch_50.t7');
model:evaluate();
os.execute("rm -rfv " .. pathOutTestData);

for d = 1, #subDirs, 1 do
	local tmpInDataPath = paths.concat(pathInTestData, subDirs[d]);
	local tmpOutDataPath = paths.concat(pathOutTestData, subDirs[d]);
	os.execute("mkdir -pv " .. tmpOutDataPath);
	local dataList = paths.dir(tmpInDataPath);
	for i = 1, #dataList, 1 do
		print(string.format("dir = %d, file = %d", d, i));
		if dataList[i]~='.' and dataList[i]~='..' then
			inputsCpu[1] = image.load(paths.concat(tmpInDataPath, dataList[i]), 3, 'float');
			inputsCpu = inputsCpu - meanData;
			inputs:copy(inputsCpu); -- copy to cuda
			local prob = model:forward(inputs);
			prob = torch.squeeze(prob):float();
			_, label = torch.max(prob,1);
			label = torch.squeeze(label);
			label = label - 1;
--			print(outputs:size());
			matio.save(paths.concat(tmpOutDataPath, dataList[i]:sub(1, dataList[i]:len()-4) .. '.mat'), {prob = prob, label = label});
		end
	end
end
