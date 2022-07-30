function rankMat=rankTransform_equalsStayEqual(mat,scale01)

% transforms the matrix mat by replacing each element by its rank in the
% distribution of all its elements.
% NaNs are ignored in this process.

if ~exist('scale01','var'), scale01=1; end;

nonNan_LOG=~isnan(mat);
set=mat(nonNan_LOG);

[sortedSet, sortedIs]=sort(set);

rankMat=nan(size(mat));
nonNan_IND=find(nonNan_LOG);
rankMat(nonNan_IND(sortedIs))=1:numel(set);

if scale01==1
    % scale into [0,1]
    rankMat=(rankMat-1)/(numel(set)-1);
elseif scale01==2
    % scale into ]0,1[
    % (best representation of a uniform distribution between 0 and 1)
    rankMat=(rankMat-.5)/numel(set);
end


%% fix artefactual differences
uniqueValues=unique(mat(nonNan_LOG));

for uniqueValueI=1:numel(uniqueValues)
    cValueEntries_LOG=mat==uniqueValues(uniqueValueI);
    rankMat(cValueEntries_LOG)=mean(rankMat(cValueEntries_LOG));
end


