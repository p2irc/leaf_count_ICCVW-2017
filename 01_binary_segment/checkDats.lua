require 'torch';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default

--local rgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/rgb/';
local dataPath = '/media/aich/DATA/leaf_seg_par/rgb';

local outDataPath = '/media/aich/DATA/leaf_seg_par_dat/rgb';

for i = 1, 90, 1 do
	tmpPath1 = paths.concat(dataPath, tostring(i));
	tmpPath2 = paths.concat(outDataPath, tostring(i));

	local nf1 = #paths.dir(tmpPath1) - 2;
	local nf2 = #paths.dir(tmpPath2) - 2;
	print(string.format("Directory = %d, Images = %d, Dats = %d", i, nf1, nf2));

	if nf1 ~= nf2 then
		print(string.format("directory mismatch = %d", i));
	end		
end

