require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

cudnn.benchmark = true;

local features = nn.Sequential()
	:add(cudnn.SpatialConvolution(3, 32, 9, 9, 1, 1, 0, 0))  -- 32 x 440 x 440
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(32, 32, 9, 9, 1, 1, 0, 0))  -- 32 x 432 x 432
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 32 x 216 x 216

	:add(cudnn.SpatialConvolution(32, 64, 9, 9, 1, 1, 0, 0))  -- 64 x 208 x 208
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(64, 64, 9, 9, 1, 1, 0, 0))  -- 64 x 200 x 200
	:add(cudnn.SpatialCrossMapLRN(5))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 64 x 100 x 100


	-- conv layer 2
	:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 0, 0))  -- 128 x 96 x 96
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(128, 128, 5, 5, 1, 1, 0, 0))  -- 128 x 92 x 92
	:add(cudnn.SpatialCrossMapLRN(10))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 128 x 46 x 46

	-- conv layer 3
	:add(cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 0, 0)) -- 256 x 42 x 42
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 0, 0)) -- 256 x 38 x 38
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 0, 0)) -- 256 x 34 x 34
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- 256 x 17 x 17

	-- conv layer 4
	:add(cudnn.SpatialConvolution(256, 512, 5, 5, 1, 1, 0, 0)) -- 512 x 13 x 13
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 5, 5, 1, 1, 0, 0)) -- 512 x 9 x 9
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 5, 5, 1, 1, 0, 0)) -- 512 x 5 x 5
	:add(cudnn.SpatialCrossMapLRN(15))
	:add(cudnn.ReLU())
	:add(cudnn.SpatialConvolution(512, 512, 5, 5, 1, 1, 0, 0)) -- 512 x 1 x 1
	:add(cudnn.ReLU());

local regressor = nn.Sequential()
	:add(nn.View(512 * 1 * 1))
	:add(nn.Linear(512, 512))
	:add(nn.Linear(512, 512))
	:add(nn.Linear(512, 1));

model = nn.Sequential():add(features):add(regressor);
print("Starting Conv layer initialization ... ");
for i = 1, model:size(), 1 do
	if (torch.type(model:get(i)) == 'cudnn.SpatialConvolution') or
			(torch.type(model:get(i)) == 'nn.SpatialConvolution') then
		-- initialize with 'Xavier'
		print(string.format("Intializing layer = %d", i));
		model:get(i):init('weight', nninit.xavier, {dist='normal', gain=1.0});
	end
end

model:cuda();

criterion = nn.SmoothL1Criterion();
criterion:cuda();

print(model);
print(criterion);
