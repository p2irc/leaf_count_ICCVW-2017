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
imFileName = 'A3_plant030_rgb.png';
map_list = {[1]=3, [2]=6, [3]=10, [4]=13, [5]=17, [6]=20,
				[7]=24, [8]=27, [9]=30, [10]=34, [11]=37, [12]=40, [13]=42 };

model = torch.load('model_count_reducer_02_ablation_epoch_31.t7');
model = model.modules[1]; -- only the features sub-model
model:cuda();
--print(model);

im1 = image.load(imFileName, 3, 'float');                                            
im1 = im1:cuda();

model:forward(im1);

for i = 1, #map_list, 1 do
	print(i);
	out=model.modules[map_list[i]]['output']:float();
	print(out:size());
	matio.save('noseg_' .. tostring(i) .. '.mat', out);
end

