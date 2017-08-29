require 'torch';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default

--local rgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/rgb/';
local rgbPath = '/media/aich/DATA/leaf_seg_par_dat/rgb';
local fgPath = '/media/aich/DATA/leaf_seg_par_dat/fg';

dirList = {}
fileList = {}

for i = 1, 90, 1 do
	tic = torch.tic();
	tmpPath1 = paths.concat(rgbPath, tostring(i));
	tmpPath2 = paths.concat(fgPath, tostring(i));
	local numFiles = #paths.dir(tmpPath1) - 2;
	for j = 1, numFiles, 1 do
		print(i, j);
		local rgb = torch.load(paths.concat(tmpPath1, tostring(j) .. '.dat'));
		local fg = torch.load(paths.concat(tmpPath2, tostring(j) .. '.dat'));
		if (rgb:size(2) ~= fg:size(1)) or (rgb:size(3) ~= fg:size(2)) then
			dirList[#dirList + 1] = i;
			fileList[#fileList + 1] = j;
		end
	end
	print(string.format("dir = %d, time = %f", i, torch.toc(tic)));
end

torch.save('inconsDir.dat', dirList);
torch.save('inconsFile.dat', fileList);
