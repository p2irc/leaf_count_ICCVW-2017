function applyLabelConsistency(AR_path, GT_path)
% Update labels in algorithmic result leaf segmentations, to match as much as
% possible the labels used in the ground truth.
% Labeled images are written in a new subfolder 'updated' of AR_path.
%
% Input:
%    AR_path - path to folder containing the images to relabel
%    GT_path - path to folder containing the ground truth label images
%
% Output:
%    [new files are created]
%
% Example usage:
%    AR_path = fullfile('data', 'AR', 'Pape_J_results', 'A1');
%    GT_path = fullfile('data', 'GT', 'A1');
%    applyLabelConsistency(AR_path, GT_path)

AR_path_updated = fullfile(AR_path, 'updated');
if exist(AR_path_updated, 'dir')
    fprintf('Removing old ''updated'' directory...\n')
    rmdir(AR_path_updated,'s')
end
mkdir(AR_path_updated)
AR_filenames = dir(fullfile(AR_path, '*_label.png'));
for i = 1:length(AR_filenames)
    AR_label = imread(fullfile(AR_path, AR_filenames(i).name));
    GT_label = imread(fullfile(GT_path, AR_filenames(i).name));
    [AR_label_updated, ~] = LabelConsistency(AR_label, GT_label);
    writeIndexedPng(AR_label_updated, fullfile(AR_path_updated, AR_filenames(i).name))
end

end

function writeIndexedPng(I, filename)
% Write image to indexed PNG file using a fixed color palette.
% Currently, the palette allows for background and up to 28 leaves/labels.

palette = [...
    0,0,0;
    252,233,79;
    114,159,207;
    239,41,41;
    173,127,168;
    138,226,52;
    233,185,110;
    252,175,62;
    211,215,207;
    196,160,0;
    32,74,135;
    164,0,0;
    92,53,102;
    78,154,6;
    143,89,2;
    206,92,0;
    136,138,133;
    237,212,0;
    52,101,164;
    204,0,0;
    117,80,123;
    115,210,22;
    193,125,17;
    245,121,0;
    186,189,182;
    136,138,133;
    85,87,83;
    46,52,54;
    238,238,236]/255;

imwrite(I, palette, filename, 'png');

end