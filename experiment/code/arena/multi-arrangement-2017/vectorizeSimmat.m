function simmat=vectorizeSimmat(simmat)
% converts simmat to lower-triangular form (row vector)
% (or leaves it in that form)

if size(simmat,1)==size(simmat,2)
    simmat(logical(eye(size(simmat))))=0; % fix diagonal: zero by definition
    simmat=squareform(simmat);
end