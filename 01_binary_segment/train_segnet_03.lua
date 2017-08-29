require 'paths';
require 'optim';
require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
local Threads = require 'threads';
Threads.serialization('threads.sharedserialize');

torch.setdefaulttensortype('torch.FloatTensor'); -- set default

local MAX_IMG_VAL = 255;

local outModelFile = "model_segnet_v1";

os.execute("mkdir logs"); -- create log directory from terminal

inSize, inDim, outSize = 224, 3, 224;
local numEpochs = 50
local batchSize = 16
require './models/model_segnet_v1_pretrained.lua';
-- require './models/model_segnet_v1.lua';
model:training();

local pathTrainData = '/media/aich/DATA/lcc_finetune_nobox_dat/rgb';
local pathTrainLabels = '/media/aich/DATA/lcc_finetune_nobox_dat/fg';

-- get mean estimate for of the training dataset
local meanData = torch.load("mean_finetune_nobox_3chn.dat")/MAX_IMG_VAL;

local numThreads = 8; -- cpu thread

local optDataThreads = { -- options for data loading threads
	manualSeed = 72, -- 72,
	numThreads = numThreads,
	inputSize = 224,
	inputSize = inSize,
	dimension = inDim,
	meanData = meanData,
	maxRgbVal = MAX_IMG_VAL
};

local printVal = true; -- print statistics or not
-- initialize performance parameters
local precision_epoch, recall_epoch, accuracy_epoch, loss_epoch;
-- specify log file
trainLogger = optim.Logger('./logs/train_segnet_01_part3.log');

local optimState = { -- for sgd
	learningRate = 0.01,
	learningRateDecay = 0.0,
	momentum = 0.9,
	dampening = 0.0,
	weightDecay = 0.0001
};

--[[
local optimState = { -- for sgd
	learningRate = 0.0005,
	learningRateDecay = 0.0,
	momentum = 0.8,
	dampening = 0.0,
	weightDecay = 0.0001
};
]]--

-- =================== System inputs end here ==============================

-- GPU inputs (preallocate)
local inputsCpu = torch.Tensor(batchSize, inDim, inSize, inSize) -- conversion from byte to float
local labelsCpu = torch.CharTensor(batchSize, outSize, outSize) -- max value is 2, char sufficient
local inputs = torch.CudaTensor(batchSize, inDim, inSize, inSize) -- float
local labels = torch.CudaTensor(batchSize, outSize, outSize) -- float

local numSubDirs = #paths.dir(pathTrainData) - 2; -- number of subdirectories
local tmpNumBatches; -- needed to print statistics inside trainBatch()

function train()

	donkeys = Threads( -- initializing parallel threads
		optDataThreads.numThreads,
		function()
			require 'torch';
		end,
		function(idx)
			options = optDataThreads; -- pass to all donkeys via upvalue
			tid = idx;
			local seed = options.manualSeed + idx;
			torch.manualSeed(seed);
			print(string.format('Starting donkey with id: %d seed: %d', tid, seed));
		end
	);

	for epoch = 14, numEpochs, 1 do
		local time_epoch = torch.Timer();
		precision_epoch = 0;
		recall_epoch = 0;
		accuracy_epoch = 0;
		loss_epoch = 0;
		local numBatches = 0;
		for dirSub = 1, numSubDirs, 1 do
			local tmpPathTrainData = paths.concat(pathTrainData, tostring(dirSub));
			local tmpPathTrainLabels = paths.concat(pathTrainLabels, tostring(dirSub));
			local tmpNumFiles = #paths.dir(tmpPathTrainData) - 2;
			tmpNumBatches = torch.ceil(tmpNumFiles/ batchSize);
			numBatches = numBatches + tmpNumBatches;
			local batch_st;
			local batch_ed = 0;
        	for batch = 1, tmpNumBatches, 1 do
				-- print statistics after every 50 batches
				if batch % 50 == 1 then
					printVal = true;
				end

				if batch ~= tmpNumBatches then
					batch_st = batch_ed + 1;
					batch_ed = batch_ed + batchSize;
				else
					batch_ed = tmpNumFiles;
					batch_st = tmpNumFiles - batchSize + 1;
				end

				----------- load files for a single batch --------------
-- 				local jobDone = 0 -- this must be local
				for fileId = batch_st, batch_ed, 1 do
					donkeys:addjob(
						function()
			--				im = image.load(dataPath .. tostring(batch) .. '.png', 3, 'float');
			--				X[batch] = image.crop(im, 1, 1, 225, 225);

							-- read and convert types, type conversion is not automatic
							local placeId = fileId - batch_st + 1;
							inputsCpu[placeId] = torch.load(paths.concat(tmpPathTrainData, tostring(fileId) .. '.dat')):float();
							labelsCpu[placeId] = torch.load(paths.concat(tmpPathTrainLabels, tostring(fileId) .. '.dat')):char();
							-- normalize range [0,1] from [0,255] and subtract mean
							inputsCpu[placeId] = (inputsCpu[placeId]/options.maxRgbVal) - options.meanData;

							collectgarbage();
							collectgarbage();
							return __threadid;
						end,
						function(id)
--							print(string.format("loaded file %d (ran on thread ID %d)", batch, id));
--							jobDone = jobDone + 1
						end
					); -- end of donkeys:addjob

				end -- end of parallel file loading
				donkeys:synchronize(); -- sync threads to complete loading all files
--				print(string.format('%d jobs done', jobDone));
				------------ loading single batch completed --------------
--				assert((labelsCpu:eq(1):sum() + labelsCpu:eq(2):sum())/labelsCpu:numel() == 1, 'other values exist');
				trainBatch(epoch, dirSub, batch); -- train loaded batch in cuda

			end -- end of training single subdirectory
		end -- end of training subdirectories

		-- save model after each epoch
		precision_epoch = precision_epoch/numBatches;
		recall_epoch = recall_epoch/numBatches;
		accuracy_epoch = accuracy_epoch/numBatches;
		loss_epoch = loss_epoch/numBatches;
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
			['Precision '] = precision_epoch,
			[' Recall '] = recall_epoch,
			[' Accuracy '] = accuracy_epoch,
			[' Loss(SCE) '] = loss_epoch,
	   	}
	   	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f,\t'
                          .. 'Loss: %.2f, \t '
                          .. 'Precision:%.2f, \t'
                          .. 'Recall:%.2f, \t'
                          .. 'Accuracy: %.2f, \t',
                       epoch, time_epoch:time().real, loss_epoch, precision_epoch, recall_epoch, accuracy_epoch))
		collectgarbage();

	end -- end of training epochs

	donkeys:terminate();
end
---------------- end of function train() ------------------

local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters = model:getParameters();
--------------------------------------------------------
function trainBatch(epoch, dirSub, batch)
	cutorch.synchronize()
	collectgarbage()
	local dataLoadingTime = dataTimer:time().real
	timer:reset()
	-- transfer over to GPU
    inputs:copy(inputsCpu)
    labels:copy(labelsCpu)
--	assert((labels:eq(1):sum() + labels:eq(2):sum())/labels:numel() == 1, 'other values exist');
	local err, outputs
	feval = function(x)
--		local tic = torch.tic()
		model:zeroGradParameters()
--		criterion:zeroGradParameters()
--		print('zero grad parameter time = ' .. torch.toc(tic));
--		local tic = torch.tic()
		outputs = model:forward(inputs);
--		print(torch.type(outputs))
--		print('model forward pass time = ' .. torch.toc(tic));
--		local tic = torch.tic()
		err = criterion:forward(outputs, labels);
--		print('criterion forward pass time = ' .. torch.toc(tic));
--		local tic = torch.tic()
		local gradOutputs = criterion:backward(outputs, labels);
--		print('criterion backward pass time = ' .. torch.toc(tic));
--		local tic = torch.tic()
		model:backward(inputs, gradOutputs)
--		print('model backward pass time = ' .. torch.toc(tic));
		return err, gradParameters
	end

--	local tic = torch.tic()
--	print('Parameter type =' .. torch.type(parameters))
--	optim.adam(feval, parameters, optimState)
--	print('optimization time = ' .. torch.toc(tic));
	optim.sgd(feval, parameters, optimState)
	cutorch.synchronize();

--	local tic = torch.tic()
	_, maxInd = torch.max(outputs, 2);
	maxInd = maxInd:type(torch.type(labels));
--	print('max index retrieval time = ' .. torch.toc(tic));
--	local tic = torch.tic()
	maxInd = torch.squeeze(maxInd); -- actual output
--	print('max matrix squeeze time = ' .. torch.toc(tic));

	-- get FN and FP from A-T
--	local tic = torch.tic()
	local tmp = maxInd - labels;
	local false_neg = tmp:eq(-1):sum();
	local false_pos = tmp:eq(1):sum();
	-- get TP and TN from A+T
	tmp = maxInd + labels;
--	print('TP FP TN FN computation time = ' .. torch.toc(tic));
	local true_neg = tmp:eq(2):sum();
	local true_pos = tmp:eq(4):sum();
	local precision = true_pos/(true_pos + false_pos);
	local recall = true_pos/(true_pos + false_neg);
	local acc = (true_pos + true_neg)/labels:numel();
	precision_epoch = precision_epoch + precision;
	recall_epoch = recall_epoch + recall;
	accuracy_epoch = accuracy_epoch + acc;
	loss_epoch = loss_epoch + err;
	if printVal == true then
		-- print information
		print(('Epoch: [%d], Subdirectory: [%d], Batch: [%d/%d]\tTime %.3f, Error(SCE) %.4f, Precision %.4f, Recall %.4f, Accuracy %.4f, DataLoadingTime %.3f'):format(epoch, dirSub, batch, tmpNumBatches, timer:time().real, err, precision, recall, acc, dataLoadingTime));
		printVal = false;
	end
   	dataTimer:reset();
end
-------------------------------------------------------
