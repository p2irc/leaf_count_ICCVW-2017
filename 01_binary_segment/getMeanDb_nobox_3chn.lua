require 'torch';
require 'paths';
require 'image';

torch.setdefaulttensortype('torch.FloatTensor');

local SIZE_IMG = 224;
local inDim = 3;

local dataPath = '/media/aich/DATA/databases/leaf_cvppp2017/lcc_original/rgb';
local imgList = paths.dir(dataPath);
local mult = 1/(#imgList - 2);
local mean_image_3chn = torch.zeros(3);
local mean_mat = torch.ones(inDim, SIZE_IMG, SIZE_IMG);

for i = 1, #imgList, 1 do
	print(i);
	if (imgList[i]~='.') and (imgList[i]~='..') then
		local im = image.load(paths.concat(dataPath, imgList[i]), 3, 'byte'):float();
		mean_image_3chn[1] = mean_image_3chn[1] + mult * im[1]:mean();
		mean_image_3chn[2] = mean_image_3chn[2] + mult * im[2]:mean();
		mean_image_3chn[3] = mean_image_3chn[3] + mult * im[3]:mean();
	end
end

print(mean_image_3chn);

mean_mat[{ {1}, {}, {} }] = mean_mat[{ {1}, {}, {} }] * mean_image_3chn[1];
mean_mat[{ {2}, {}, {} }] = mean_mat[{ {2}, {}, {} }] * mean_image_3chn[2];
mean_mat[{ {3}, {}, {} }] = mean_mat[{ {3}, {}, {} }] * mean_image_3chn[3];

print(mean_mat[1]:max(), mean_mat[1]:min());
print(mean_mat[2]:max(), mean_mat[2]:min());
print(mean_mat[3]:max(), mean_mat[3]:min());

torch.save('mean_finetune_nobox_3chn.dat', mean_mat);
