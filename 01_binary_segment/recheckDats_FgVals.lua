require 'torch';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default

--local rgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/rgb/';
local rgbPath = '/media/aich/DATA/leaf_seg_par_dat/rgb';
local fgPath = '/media/aich/DATA/leaf_seg_par_dat/fg';

lowDirList = {}
lowFileList = {}

for i = 1, 90, 1 do
	tic = torch.tic();
	local tmpPath = paths.concat(fgPath, tostring(i));
	local numFiles = #paths.dir(tmpPath) - 2;
	for j = 1, numFiles, 1 do
		print(i, j);
		local fg = torch.load(paths.concat(tmpPath, tostring(j) .. '.dat'));
		if (fg:min() == 0) and (fg:max()==1) then
			lowDirList[#lowDirList + 1] = i;
			lowFileList[#lowFileList + 1] = j;
			fg = fg + 1;
			torch.save(paths.concat(tmpPath, tostring(j) .. '.dat'), fg);
		end
		
	end
	print(string.format("dir = %d, time = %f", i, torch.toc(tic)));
end

torch.save('lowDir.dat', lowDirList);
torch.save('lowFile.dat', lowFileList);
