-----
### Training SegNet

-----
#### Generate training dataset (~0.8M) at stage 01 and train
-----
* separateSegData.m
* genBlurData.m
* genRotBboxData\_parallel.m (OR) genRotBboxData.m
* adjustImgSize.m
* convert\_png\_to\_dat\_adjImgSize.lua
    -> also computes means and saves in directory\_mean.dat and total\_mean.dat
* checkDats.lua -> remove the mismatched files
* recheckDatSize.lua
* recheckDats\_FgVals.lua
* recheckDats.lua -> remove the inconsistent files
* train\_segnet\_01.lua -> trains for 5 (1-5) epochs in the first stage
    -> change variable numThreads specifying the number of threads to load batches.
* main.lua _(change inside)_

-----
#### Generate training dataset (~0.3M) at stage 02 and train
-----
* sepLccBasicData.m
* getBboxImage.m
* genFineTuneBinSegData.m
* convert\_png\_to\_dat\_finetune.lua
* getMeanDb.lua
* train\_segnet\_02.lua -> trains for 8 (6-13) epochs in the first stage
    -> change variable numThreads specifying the number of threads to load batches.
* main.lua _(change inside)_

-----
#### Generate training dataset (~0.19M) at stage 03 and train
-----
* genFineTuneBinSegData\_nobox.m
* convert\_png\_to\_dat\_finetune\_nobox.lua
* getMeanDb\_nobox\_3chn.lua
* train\_segnet\_03.lua -> trains for 37 (14-50) epochs in the first stage
    -> change variable numThreads specifying the number of threads to load batches.
* main.lua _(change inside)_

-----

-----
#### Generate binary segmentations for training images (after training)
-----
* sepTrainDataBinSeg.m
* genTrainDataBinSeg\_nobox.m
* genTrainBinSeg\_nobox.lua
* mergeTrainDataBinSeg\_nobox.m
    -> generates bs\_sum\_plain\_nobox directory used as final segmentation
* genFillTrain\_nobox.m _(not necessary)_
* makeTrainCompositeBinSeg\_nobox.m
* makeFillCompTrain\_nobox.m _(not necessary)_

-----
#### Generate binary segmentations for test images (after training)
-----
* sepTestDataBinSeg.m
* genTestDataBinSeg\_nobox.m
* genTestBinSeg\_nobox.lua
* mergeTestDataBinSeg\_nobox.m
    -> generates bs\_sum\_plain\_nobox directory used as final segmentation
* genFillTest\_nobox.m _(not necessary)_
* makeTestCompositeBinSeg\_nobox.m
* makeFillCompTest\_nobox.m _(not necessary)_

-----
#### Get Precision-Recall
-----
* getPrecRecall\_train.m _(training set)_
* getPrecRecall.m _(test set)_

-----
#### Get G-BR Threshold _(only for exploratory analysis, not necessary)_
-----
* getThresh_gbr.m

-----
#### Get the hard training images _(only for exploratory analysis, not necessary)_
-----
* sepHardTrainDataBinSeg.m
