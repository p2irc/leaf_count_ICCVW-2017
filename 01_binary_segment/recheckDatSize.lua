require 'torch';
require 'paths';
torch.setdefaulttensortype('torch.FloatTensor'); -- set default
local MIN_SIZE = 224; -- minimum size of each image
local numThreads = 12;

--local rgbPath = '/media/aich/DATA/databases/leaf_cvppp2017/seg_blur/rgb/';
local dataPath = '/media/aich/DATA/leaf_seg_par_dat/';
local rgbPath = './rgb';
local fgPath = './fg';

local numDirs = #paths.dir(paths.concat(dataPath, rgbPath)) - 2;
local dirMean = torch.zeros(numDirs, 1); -- directory wise mean estimate

local rgbPath = paths.concat(dataPath, rgbPath);
local fgPath = paths.concat(dataPath, fgPath);
local outRgbPath = paths.concat(outDataPath, outRgbPath);
local outFgPath = paths.concat(outDataPath, outFgPath);
os.execute("mkdir -pv " .. outRgbPath);
os.execute("mkdir -pv " .. outFgPath);

local Threads = require 'threads';
Threads.serialization('threads.sharedserialize');
--Threads.serialization('threads.serialize');

donkeys = Threads(
	numThreads,
	function()
		require 'torch';
		require 'paths';
	end,
	function(idx)
		local rgbPath = rgbPath;
		local fgPath = fgPath;
		local outRgbPath = outRgbPath;
		local outFgPath = outFgPath;
		torch.setdefaulttensortype('torch.FloatTensor');
	    tid = idx;
	    print(string.format('Starting donkey with id: %d', tid));
	end
);

local jobDone = 0 -- this must be local
for d = 1, numDirs, 1 do
	donkeys:addjob(
		function()
			local rgbPath = paths.concat(rgbPath, d);
			local fgPath = paths.concat(fgPath, d);
			local outRgbPath = paths.concat(outRgbPath, d);
			local outFgPath = paths.concat(outFgPath, d);

			local fileList = paths.dir(rgbPath);

			local numFiles = #fileList - 2;
			local count = 0;
			local r_size, c_size, r_lb, r_ub, c_lb, c_ub;
			for fid = 1, #fileList, 1 do
				pcall( function()
					if fid % 50 == 1 then
						print(string.format("dir = %d, file = %d, thread = %d", d, fid, __threadid));
						collectgarbage();
						collectgarbage();
					end
					if (fileList[fid] ~= '.') and (fileList[fid] ~= '..') then
						local tmpFgPath = paths.concat(fgPath, fileList[fid]);
						local tmpRgbPath = paths.concat(rgbPath, fileList[fid]);
						local gt = torch.load(tmpFgPath);
						gt = torch.squeeze(gt);
						local im = torch.load(tmpRgbPath);
						-- save individual means
						dirMean[d] = dirMean[d] + (im:sum()/im:numel())/numFiles;

						if (gt:size(1) < MIN_SIZE) or (gt:size(2) < MIN_SIZE) then
							if gt:size(1) < MIN_SIZE then
								r_lb = torch.floor((MIN_SIZE - gt:size(1) )/2) + 1;
								r_ub = r_lb + gt:size(1) - 1;
								r_size = MIN_SIZE;
							else
								r_lb = 1;
								r_ub = gt:size(1);
								r_size = r_ub;
							end

							if gt:size(2) < MIN_SIZE then
								c_lb = torch.floor((MIN_SIZE - gt:size(2) )/2) + 1;
								c_ub = c_lb + gt:size(2) - 1;
								c_size = MIN_SIZE;
							else
								c_lb = 1;
								c_ub = gt:size(2);
								c_size = c_ub;
							end

							print(string.format("r_lb=%d, r_ub=%d, c_lb=%d, c_ub=%d", r_lb, r_ub, c_lb, c_ub));
							local im_new = torch.zeros(3, r_size, c_size):byte() ;
							local gt_new = torch.zeros(r_size, c_size):byte() ;


							im_new[{ {}, {r_lb, r_ub}, {c_lb, c_ub} }] = im;
							gt_new[{ {r_lb, r_ub}, {c_lb, c_ub} }] = gt;

							
							
							gt_new = gt_new + 1;
							-- save in the output locations (.dat)
							count = count + 1;
							local fileName = tostring(count) .. '.dat';
							torch.save(paths.concat(outRgbPath, fileName), im_new);
							torch.save(paths.concat(outFgPath, fileName), gt_new);

							collectgarbage();
							collectgarbage();
						else
							gt = gt + 1;
							count = count + 1;
							local fileName = tostring(count) .. '.dat';
							torch.save(paths.concat(outRgbPath, fileName), im);
							torch.save(paths.concat(outFgPath, fileName), gt);
						end
					end
				end )
			end


			collectgarbage();
			collectgarbage();
			return __threadid, (#fileList-2), d;
		end,
		function(id, numFiles, dirId)
			print(string.format("Finished directory = %d, #files = %d (ran on thread ID %d)", dirId, numFiles, id));
			jobDone = jobDone + 1
		end
	);
end

donkeys:synchronize();
print(string.format('%d jobs done', jobDone));
donkeys:terminate();

torch.save('./directory_dat_mean.dat', dirMean);
local mean_0_255 = dirMean:mean();
print(string.format("Estimated mean [0, 255] = %f", mean_0_255));
-- torch.save('./total_mean.dat', mean_0_255);


