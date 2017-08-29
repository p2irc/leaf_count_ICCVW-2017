function [outLabel,asL] = LabelConsistency(inLabel,gtLabel)
% inLabel: label image (2D matrix) to be evaluated. Labels are assumed to be consecutive numbers.
% gtLabel: ground truth label image (2D matrix). Labels are assumed to be consecutive numbers.
% overlap function used: Dice score
% ASSUMPTIONS. gtLabel is the "fixed" image and the one that it will used
% to infer "correct" cluster assignments
%  
%
% For the original gtLabel, labels corresponding to each other need to
% be known in advance. Here we simply take the best matching label from 
% gtLabel in each comparison. We do not make sure that a label from gtLabel
% is used only once. Better measures may exist. Please enlighten me if I do
% something stupid here...
% Outputs:
% vol_out the inLabel image with new label IDs 
% non matched labels get a
% number >MM assuming that gtLabel has <MM labels.
% asL a list that is showing the mapping of cluster IDs before after....
% there is a flag verbose that controls what is displayed on screen default is 1

% minimum threshold for a dice match
t = 0.2;
% MM = 30;
KK = -100; % just a check value so i dont reuse

% check if label images have same size
if(max(size(inLabel)~=size(gtLabel)))
    outLabel = [];
    disp('ooops not the same size')
    return
end

verbose = 1; % flag for verbose messages, set other than 1 to suspend them

%assuming 0 is the background do not process the background
% NOTICE we assume consecutive LABELS, starting from 0 for background!
% if they are not we have I wrote a new function to correct things
% here!!!

inLabel = CheckConsecutive(inLabel);


maxInLabel = max(inLabel(:)); % maximum label value in inLabel
minInLabel = max([min(inLabel(:)), 1]); % minimum label value in inLabel
maxGtLabel = max(gtLabel(:)); % maximum label value in gtLabel
minGtLabel = max([min(gtLabel(:)), 1]); % minimum label value in gtLabel

if verbose == 1
    fprintf('AR has %d labels\n', maxInLabel - minInLabel)
    fprintf('GT has %d labels\n', maxGtLabel - minGtLabel)
end


outLabel = zeros(size(inLabel), 'uint8'); % initialize output
label_matrix = KK*ones(maxInLabel,maxGtLabel);

for i=minInLabel:maxInLabel; % loop all labels of inLabel
    
    for j=minGtLabel:maxGtLabel % loop all labels of gtLabel
        s = Dice(inLabel, gtLabel, i, j); % compare labelled regions
        % keep max Dice value for label i
        if(s > t) % if dice value is not greather than t ignore!
            label_matrix(i,j) = s;
        end
    end
    % for a given i label in the volume, assign it the best matching label in the gtLabel volume.
%     if sMax ==-10
%         if verbose
%             disp(sprintf('something is off and is not normal for label i=%d',i))
%         end
%     elseif sMax ==0
%         disp(sprintf('this means no good overlap was found, assigning dummy values for this label i=%d',i))
%         vol_out = (i+maxGtLabel).* (inLabel==i) + vol_out;
%     else
%         if verbose
%             disp(sprintf('supposedly everything is ok for label i=%d assigned to j_best=%d',i,j_best))
%         end
%         vol_out = j_best.* (inLabel==i) + vol_out;
%     end
%     
end

% this parts scans throught the score matrix and finds the best label assignment
asL = [];
listIn = minInLabel:maxInLabel;
% listGt = [minGtLabel:maxGtLabel];

done = 0 ;

M = label_matrix;
% process the easy stuff first!
while done == 0
    % start from global maximum of matrix and move to the next
    cmax = max(M(:));
    if cmax == KK % check if maximum is the reserved value. if yes then stop.
        % done = 1;
        break;
    end
    
    [i,j] = find(M==cmax);
    
    % check if more than 1
    if length(i)>1
        i=i(1); % greedily just pick the first
        j=j(1);
    end
    
    % found a match
    
    asL = [asL; [i j]];
    
    % cancel parts of M matrix
    M(i,:) = KK;
    M(:,j) = KK;
end

% so lets check current state

if size(asL,1)==maxInLabel
    % all great done
    if verbose==1
        fprintf('found enough number of matches\n')
    end
elseif size(asL,1)<maxInLabel
    % oops somebody is not assigned
    JJ = setdiff(listIn,asL(:,1)');
    
    % for these labels give them a new assignment, starting from the first unused label in gtLabel
    newL = maxGtLabel+1:(maxGtLabel+length(JJ));
    asL = [asL; [JJ' newL']];
    
    if verbose==1
        fprintf('I did not find enough number of matches, I am creating %d new one(s)\n', length(JJ))
        disp(asL)
    end
else
    disp('bigger? how can it be?')
end

% everyone should have a label
% check any way
if size(asL,1)~=maxInLabel
    disp('oops error smth went wrong')
    outLabel =[];
    
    return;
end

% apply the index mapping
for i = 1:size(asL,1)
    outLabel(inLabel == asL(i,1)) = asL(i,2);
end
end

function out = Dice(inLabel, gtLabel, i, j)
% calculate Dice score for the given labels i and j

inMask = (inLabel==i); % find region of label i in inLabel
gtMask = (gtLabel==j); % find region of label j in gtLabel
out = 2*nnz(gtMask & inMask)/(nnz(gtMask) + nnz(inMask)); % Dice score
end

function out = CheckConsecutive(in)
%ensures that labeling is consecutive

labels = unique(in(:));
consec = (0:(length(labels)-1))';

if ~isequal(labels,consec)
    % ouch things are not consecutive ... relabel
    asL = [labels consec];
    out = zeros(size(in));  %%% always assume that 0 is background
    for i = 1:length(labels)
        ii = double(asL(i,1));
        j_best = double(asL(i,2));
        out = j_best.* double(double(in)==ii) + out;
    end
    
else
    out = in;
end
end