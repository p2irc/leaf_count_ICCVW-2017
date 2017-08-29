require 'paths';
require 'optim';
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local matio = require 'matio';

torch.setdefaulttensortype('torch.FloatTensor'); -- set default


local outModelFile = "model_count_reducer_02_ablation";

os.execute("mkdir logs"); -- create log directory from terminal

require './models/model_count_reducer_02_ablation.lua';
--require './models/model_count_reducer_pretrained_02_ablation.lua';

model:training();
-- require './models/model_count_v1.lua';

local numEpochs = 35;
local pathTrainData = '/media/aich/DATA/databases/lcc_seg_aug_ablation_dat';
local pathTrainLabels = '/media/aich/DATA/databases/leaf_cvppp2017/train_count';
local labelFileName = 'gt_seg_aug.mat';

-- specify log file
trainLogger = optim.Logger('./logs/train_count_reducer_ablation_02_part1.log');

local optimState = { -- for sgd
	learningRate = 0.0001,
	learningRateDecay = 0.0,
	momentum = 0.9,
	dampening = 0.0,
	weightDecay = 0.0001
};

-- =================== System inputs end here ==============================

-- GPU inputs (preallocate)
local inputsCpu = torch.Tensor(1, 3, 448, 448);
local labelsCpu = torch.Tensor(1);
local inputs = torch.CudaTensor(1, 3, 448, 448);
local labels = torch.CudaTensor(1);
local numFiles = #paths.dir(pathTrainData) - 2;
local weight_mult = 1/numFiles;
local overestim_epoch, underestim_epoch, loss_epoch;
local printVal = false;
function train()

	-- load all labels from mat file
	local dataLabels = matio.load(paths.concat(pathTrainLabels, labelFileName), 'g_GtAug'):float();
	print(string.format("Total Leaf Count = %d", dataLabels:sum()));
	for epoch = 1, numEpochs, 1 do
		local time_epoch = torch.Timer();
		overestim_epoch, underestim_epoch, loss_epoch = 0, 0, 0;
		local fileList = torch.randperm(numFiles); -- load different permutations in different epochs
		for i = 1, numFiles, 1 do
			if i%50 == 1 then
				printVal = true;
			end
			inputsCpu = torch.load(paths.concat(pathTrainData, tostring(fileList[i]) .. '.dat'));
			labelsCpu = dataLabels[fileList[i]];
			trainBatch(epoch, i, fileList[i]); -- train loaded batch in cuda
		end

		-- clear the intermediate states in the model before saving to disk
	   	-- this saves lots of disk space
	   	model:clearState();
		local fullModelFile = "./models/" .. outModelFile .. "_epoch_" .. tostring(epoch) .. ".t7";
	   	torch.save(fullModelFile , model) -- save model after each epoch

	   	trainLogger:add{
			['Epoch '] = epoch,
			['LearningRate '] = optimState.learningRate,
			['Momentum '] = optimState.momentum,
			['WeightDecay '] = optimState.weightDecay,
			['TotalCount '] = dataLabels:sum(),
			[' Overestimate '] = overestim_epoch,
			[' Underestimate '] = underestim_epoch,
			[' Loss(SL1) '] = loss_epoch,
	   	}
	   	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f,\t'
                          .. 'Loss: %.2f, \t '
                          .. 'Underestimate: %.2f, \t'
                          .. 'Overestimate: %.2f, \t',
                       epoch, time_epoch:time().real, loss_epoch, underestim_epoch, overestim_epoch));
		collectgarbage();

	end -- end of training epochs
end
---------------- end of function train() ------------------

local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters = model:getParameters();
--------------------------------------------------------
function trainBatch(epoch, i, fid)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()

	--inputs:resize(inputsCpu:size()):copy(inputsCpu)
	--labels:resize(labelsCpu:size()):copy(labelsCpu)
	inputs:copy(inputsCpu);
	labels:copy(labelsCpu);

	local err, outputs
	feval = function(x)
		model:zeroGradParameters()
		outputs = model:forward(inputs);
		err = criterion:forward(outputs, labels);
		local gradOutputs = criterion:backward(outputs, labels);
		model:backward(inputs, gradOutputs)
		return err, gradParameters
	end

--	optim.sgd(feval, parameters, optimState)
	optim.adam(feval, parameters, optimState)
	cutorch.synchronize();

	local dif = outputs - labels;
	if dif[1] < 0 then
		underestim_epoch = underestim_epoch + torch.abs(dif[1]);
	elseif dif[1] > 0 then
		overestim_epoch = overestim_epoch + dif[1]
	end
	loss_epoch = loss_epoch + err * weight_mult;

	-- print information
	if printVal == true then
		print(('Epoch: [%d], File: [%d/%d], FileID: [%d]\tTime %.3f, Error(SL1) %.4f, Count(GT) %d, Difference %.4f, DataLoadingTime %.3f'):format(epoch, i, numFiles, fid, timer:time().real, err, labels[1], dif[1], dataLoadingTime));
		printVal = false;
	end
   	dataTimer:reset();
end
-------------------------------------------------------
