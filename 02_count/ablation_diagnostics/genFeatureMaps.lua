require 'torch';
require 'cutorch';
require 'cudnn';
require 'nn';
require 'cunn';
require 'image';
local matio = require 'matio';
cudnn.benchmark = true;

torch.setdefaulttensortype('torch.FloatTensor');

--imFileName = 'plant003_rgb.png';
--gtFileName = 'plant003_bs.png';
imFileName = 'A3_plant030_rgb.png';
gtFileName = 'A3_plant030_bs.png';
map_list = {[1]=3, [2]=6, [3]=10, [4]=13, [5]=17, [6]=20,
				[7]=24, [8]=27, [9]=30, [10]=34, [11]=37, [12]=40, [13]=42 };

model = torch.load('model_count_reducer_02_epoch_35.t7');
model = model.modules[1]; -- only the features sub-model
model:cuda();
--print(model);

im = image.load(imFileName, 3, 'float');                                            
gt = image.load(gtFileName, 1, 'float');
im1 = torch.cat(im, gt, 1);
im1 = im1:cuda();

model:forward(im1);

for i = 1, #map_list, 1 do
	print(i);
	out=model.modules[map_list[i]]['output']:float();
	print(out:size());
	matio.save('seg_' .. tostring(i) .. '.mat', out);
end

