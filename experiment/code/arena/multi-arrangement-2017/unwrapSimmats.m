function [simmats,nSimmats]=unwrapSimmats(simmats_struct)
% unwraps similiarity matrices in a structured array with meta data by
% extracting the similarity matrices (in square or lower triangle form) and
% lining them up along the third dimension. (if they are already in that
% format they are handed back unchanged.)

if strcmp(class(simmats_struct),'struct')
    % in struct form
    nSimmats=size(simmats_struct,2);
    if length(simmats_struct(1).simmat)==numel(simmats_struct(1).simmat)
        % in upper-triangular form
        simmats=nan(1,length(simmats_struct(1).simmat),nSimmats);
        for simmatI=1:nSimmats
            simmats(1,:,simmatI)=vectorizeSimmat(simmats_struct(simmatI).simmat);
        end
    else
        % in square form
        simmats=nan(size(simmats_struct(1).simmat,1),size(simmats_struct(1).simmat,2),nSimmats);
        for simmatI=1:nSimmats
            simmats(:,:,simmatI)=squareSimmat(simmats_struct(simmatI).simmat);
        end
    end
else
    % bare already
    simmats=simmats_struct;
    nSimmats=size(simmats,3);
end
