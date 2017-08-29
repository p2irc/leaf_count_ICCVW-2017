require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';

cudnn.benchmark = true;

model = torch.load('./models/model_count_reducer_02_ablation_epoch_5.t7');
model:cuda();

criterion = nn.SmoothL1Criterion();
criterion:cuda();

print(model);
print(criterion);
