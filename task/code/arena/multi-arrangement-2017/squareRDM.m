function RDM=squareRDM(RDM)
% converts RDM to square form
%
% if they are in square form already they are vectorized
% using the lower triangle (matlab squareform convention) and resquared,
% thus fixing inconsistencies between upper and lower triangle (the lower
% is used).

RDM=squareform(vectorizeRDM(RDM));
