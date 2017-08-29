-- load pretrained models
require 'torch'
require 'cutorch'
require 'cudnn'
require 'nn'
require 'cunn'

cudnn.benchmark = true;

--[[ 
history of weights
epoch = 1-->5, train with box and fg weight = 2.0
epoch = 6-->13, train with box and fg weight = 1.2
epoch = 14-->32, finetune with nobox and fg weight = 1.2
epoch = 33->  , train with hard cases and fg weight = 1.2
]]--

local weights = torch.Tensor(2); -- weights for loss calculation
weights[1] = 1.0; -- background
weights[2] = 1.2; -- foreground 

-- model = torch.load('./models/model_segnet_v1_epoch_13.t7'); -- finetune
model = torch.load('./models/model_segnet_v1_epoch_32.t7'); 
model:cuda();

criterion = cudnn.SpatialCrossEntropyCriterion(weights)
criterion:cuda();


