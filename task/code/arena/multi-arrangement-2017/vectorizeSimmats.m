function simmats_utv=vectorizeSimmats(simmats)
% converts set of simmats (stacked along the 3rd dimension)
% to upper-triangular form (set of row vectors)

if isstruct(simmats)
    % wrapped
    simmats_struct=simmats;
    simmats=unwrapSimmats(simmats_struct);
    
    nSimmats=size(simmats,3);
    simmats_utv=[];
    for simmatI=1:nSimmats
        simmats_utv=cat(3,simmats_utv,vectorizeSimmat(simmats(:,:,simmatI)));
    end
    
    simmats_utv=wrapSimmats(simmats_utv,simmats_struct);
else
    % bare
    nSimmats=size(simmats,3);
    simmats_utv=[];
    for simmatI=1:nSimmats
        simmats_utv=cat(3,simmats_utv,vectorizeSimmat(simmats(:,:,simmatI)));
    end
end