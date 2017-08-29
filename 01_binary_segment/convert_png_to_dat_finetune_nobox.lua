require 'torch';
require 'image';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default

local inPath = '/media/aich/DATA/databases/lcc_finetune_nobox/';
local outPath = '/media/aich/DATA/lcc_finetune_nobox_dat/';
local rgbPath = 'rgb';
local fgPath = 'fg';

local inRgbPath = paths.concat(inPath, rgbPath);
local inFgPath = paths.concat(inPath, fgPath);
local outRgbPath = paths.concat(outPath, rgbPath);
local outFgPath = paths.concat(outPath, fgPath);

local numDirs = #paths.dir(inRgbPath) - 2;

os.execute("mkdir -pv " .. outRgbPath);
os.execute("mkdir -pv " .. outFgPath);

for d = 1, numDirs, 1 do
	local tmpInRgbPath = paths.concat(inRgbPath, tostring(d));
	local tmpInFgPath = paths.concat(inFgPath, tostring(d));
	local tmpOutRgbPath = paths.concat(outRgbPath, tostring(d));
	local tmpOutFgPath = paths.concat(outFgPath, tostring(d));	
	os.execute("mkdir -v " .. tmpOutRgbPath);
	os.execute("mkdir -v " .. tmpOutFgPath);
	local numFiles = #paths.dir(tmpInRgbPath) - 2;
	for i = 1, numFiles, 1 do
		print(string.format("dir = %d, file = %d", d, i));
		local tmpImPath = paths.concat(tmpInRgbPath, tostring(i) .. '.png');
		local tmpGtPath = paths.concat(tmpInFgPath, tostring(i) .. '.png');
		local im = image.load(tmpImPath, 3, 'byte');
		local gt = torch.squeeze(image.load(tmpGtPath, 1, 'byte'));
		gt[gt:eq(255)] = 2;
		gt[gt:eq(0)] = 1;
		local tmpOutImPath = paths.concat(tmpOutRgbPath, tostring(i) .. '.dat');
		local tmpOutGtPath = paths.concat(tmpOutFgPath, tostring(i) .. '.dat');
		torch.save(tmpOutImPath, im);
		torch.save(tmpOutGtPath, gt);
	end
end
