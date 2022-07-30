function rankMat=rankTransform(mat,scale01)

% transforms the matrix mat by replacing each element by its rank in the
% distribution of all its elements

if ~exist('scale01','var'), scale01=false; end;

nonNan_LOG=~isnan(mat);
set=mat(nonNan_LOG);

[sortedSet, sortedIs]=sort(set);

rankMat=nan(size(mat));
nonNan_IND=find(nonNan_LOG);
rankMat(nonNan_IND(sortedIs))=1:numel(set);

if scale01
    rankMat=rankMat/numel(set);
end

