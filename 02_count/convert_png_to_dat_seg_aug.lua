require 'image';
require 'paths';

torch.setdefaulttensortype('torch.FloatTensor');

local pathData = '/media/aich/DATA/databases/leaf_cvppp2017/train_count/data_seg_aug';
local pathBs = '/media/aich/DATA/databases/leaf_cvppp2017/train_count/bs_seg_aug';
local pathDatFiles = '/media/aich/DATA/databases/lcc_seg_aug_dat';

os.execute("rm -rfv " .. pathDatFiles);
os.execute("mkdir -v " .. pathDatFiles);

mean_seg_aug = torch.load('mean_seg_aug.dat');

numFiles = #paths.dir(pathData) - 2;
print(numFiles);

for i = 1, numFiles, 1 do
	print(i);
	local im = image.load(paths.concat(pathData, tostring(i) .. '.png'), 3, 'float'); -- - mean_seg_aug; 
	local gt = image.load(paths.concat(pathBs, tostring(i) .. '.png'), 1, 'float');
	local data = torch.cat(im, gt, 1);
	
	torch.save(paths.concat(pathDatFiles, tostring(i) .. '.dat'), data);
end



