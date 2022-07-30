function RDM=vectorizeRDM(RDM)
% converts RDM to upper-triangular form (row vector)
% (or leaves it in that form)

if size(RDM,1)==size(RDM,2)
    RDM(logical(eye(size(RDM))))=0; % fix diagonal: zero by definition
    RDM=squareform(RDM);
end