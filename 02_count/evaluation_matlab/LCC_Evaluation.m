% Evaluation of leaf counting results for the Leaf Counting Challenge (LCC) of
% the Computer Vision Problems in Plant Phenotyping (CVPPP) workshop.
%
% Author:  Massimo Minervini
% Contact: massimo.minervini@imtlucca.it
% Version: 1.0
% Date:    02/04/2015
%
% Copyright (C) 2015 Pattern Recognition and Image Analysis (PRIAn) Unit,
% IMT Institute for Advanced Studies, Lucca, Italy.
% All rights reserved.
%
% Adapted for LCC 2017 by Hanno Scharr

function LCC_Evaluation(AR_path, GT_path)
% Entry-point function to launch the evaluation
%
% Input:
%    AR_path - Path to folder containing the submission folders (algorithmic results).
%              A submission for a dataset can be omitted, but if present then all
%              images must have a value.
%              Folder structure must be as in the following example:
%              AR/
%              +--- Doe_J_results/
%              |    +--- A1.csv
%              |    +--- A2.csv
%              |    +--- A3.csv
%              |    +--- A4.csv
%              +--- Smith_J_results/
%                   +--- A1.csv
%                   +--- A2.csv
%                   +--- A3.csv
%                   +--- A4.csv
%    GT_path - Path to folder containing the ground truth CSV files.
%              Folder structure must be as in the following example:
%              GT/
%              +--- A1.csv
%              +--- A2.csv
%              +--- A3.csv
%              +--- A4.csv
%
% Output:
%    [new .csv and .tex files are created in the AR_path folder]
%
% Example usage:
%    AR_path = fullfile('LCC_data', 'AR');
%    GT_path = fullfile('LCC_data', 'GT');
%    LCC_Evaluation(AR_path, GT_path)

AR_folders = dir(AR_path);
AR_folders = AR_folders([AR_folders.isdir]);
AR_folders = AR_folders(arrayfun(@(x) x.name(1), AR_folders) ~= '.');
for k = 1:length(AR_folders)
    results = {};
    AR_IDs_all = []; AR_counts_all = []; GT_counts_all = [];
    for expNumber = 1:5
        AR_CSV_filename = fullfile(AR_path, AR_folders(k).name, ['A' num2str(expNumber,'%d') '.csv']);
        if exist(AR_CSV_filename, 'file')
            expName = ['A' num2str(expNumber,'%d')];
            GT_CSV_filename = fullfile(GT_path, [expName '.csv']);
            if exist(GT_CSV_filename, 'file')
                fprintf('Processing ''%s'' ...\n', AR_CSV_filename)
                fid = fopen(AR_CSV_filename, 'rt');
                AR_T = textscan(fid, '%s %d', 'Delimiter', ',');
                fclose(fid);
                fid = fopen(GT_CSV_filename, 'rt');
                GT_T = textscan(fid, '%s %d', 'Delimiter', ',');
                fclose(fid);
                AR_T = checkCorrespondence(AR_T, GT_T);
                AR_IDs_all = [AR_IDs_all; AR_T{1}];
                AR_counts_all = [AR_counts_all; AR_T{2}];
                GT_counts_all = [GT_counts_all; GT_T{2}];
                results{expNumber} = evaluateResults(double(AR_T{2}), double(GT_T{2}));
                writeResultsToCSV(fullfile(AR_path, [AR_folders(k).name '_' expName '_LCC.csv']), GT_T{1}, results{expNumber})
            end
        end
    end
    results_all = evaluateResults(double(AR_counts_all), double(GT_counts_all));
    writeResultsToCSV(fullfile(AR_path, [AR_folders(k).name '_all_LCC.csv']), AR_IDs_all, results_all)
    writeResultsToLaTeX(fullfile(AR_path, [AR_folders(k).name '_LCC.tex']), results, results_all)
end

end

function AR_T = checkCorrespondence(AR_T, GT_T)
% Check the correspondance between image IDs and GT

% AR and GT must have the same length
if length(AR_T{1}) ~= length(GT_T{1})
    error('Algorithmic result and ground truth CSV files must have the same number of elements.')
end

% permute AR to match the ordering of GT, based on the first column (image ID)
if ~isequal(AR_T{1}, GT_T{1})
    [B, idx] = sort(AR_T{1});
    AR_T{1} = B;
    AR_T{2} = AR_T{2}(idx);
end

% AR and GT must have the same list of IDs
if ~isequal(AR_T{1}, GT_T{1})
    error('Algorithmic result and ground truth CSV files must include the same list of image IDs.')
end

end

function results = evaluateResults(AR, GT)
% Compute leaf counting evaluation criteria

results.N = length(GT);

results.CountDiff = AR - GT;
results.CountDiff_mean = mean(results.CountDiff);
results.CountDiff_std = std(results.CountDiff);

results.AbsCountDiff = abs(results.CountDiff);
results.AbsCountDiff_mean = mean(results.AbsCountDiff);
results.AbsCountDiff_std = std(results.AbsCountDiff);

results.CountAgreement = (AR == GT);
results.PercentAgreement = sum(results.CountAgreement)/results.N*100;

results.MSE = mean((AR - GT).^2);

end

function writeResultsToCSV(filepath, IDs, results)
% Write results in tabular format to a comma-separated values (CSV) file

fprintf('Writing results to ''%s'' (CSV) ...\n', filepath);
fid = fopen(filepath, 'w+');
% write results for each individual image
fprintf(fid, '%s, %s, %s, %s\n', 'ID', 'CountDiff', 'AbsCountDiff', 'CountAgreement');
for k = 1:results.N
    fprintf(fid, '%s, %d, %d, %d\n', IDs{k}, results.CountDiff(k), results.AbsCountDiff(k), results.CountAgreement(k));
end
% write overall statistics
fprintf(fid, '\n');
fprintf(fid, 'CountDiff (mean), %.2f\n', results.CountDiff_mean);
fprintf(fid, 'CountDiff (std), %.2f\n', results.CountDiff_std);
fprintf(fid, 'AbsCountDiff (mean), %.2f\n', results.AbsCountDiff_mean);
fprintf(fid, 'AbsCountDiff (std), %.2f\n', results.AbsCountDiff_std);
fprintf(fid, 'PercentAgreement, %.1f\n', results.PercentAgreement);
fprintf(fid, 'MSE, %.2f\n', results.MSE);

fclose(fid);

end

function writeResultsToLaTeX(filepath, results, results_all)
% Write results in LaTeX tabular format and save to a .tex file

fprintf('Writing results to ''%s'' (LaTeX) ...\n', filepath);
fid = fopen(filepath, 'w+');

fprintf(fid, '\\begin{tabular}{lcccc}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, ' & CountDiff & AbsCountDiff & PercentAgreement [\\%%] & MSE \\\\\n');
fprintf(fid, '\\hline\n');
% write results for each dataset
for expNumber = 1:length(results)
    if isempty(results{expNumber})
        continue
    end
    fprintf(fid, 'A%d & %.2f(%.2f) & %.2f(%.2f) & %.1f & %.2f \\\\\n', ...
        expNumber,...
        results{expNumber}.CountDiff_mean, results{expNumber}.CountDiff_std, ...
        results{expNumber}.AbsCountDiff_mean, results{expNumber}.AbsCountDiff_std, ...
        results{expNumber}.PercentAgreement, ...
        results{expNumber}.MSE);
end
fprintf(fid, '\\hline\n');
% write overall results
fprintf(fid, 'All & %.2f(%.2f) & %.2f(%.2f) & %.1f & %.2f \\\\\n', ...
    results_all.CountDiff_mean, results_all.CountDiff_std, ...
    results_all.AbsCountDiff_mean, results_all.AbsCountDiff_std, ...
    results_all.PercentAgreement, ...
    results_all.MSE);
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');

fclose(fid);

end