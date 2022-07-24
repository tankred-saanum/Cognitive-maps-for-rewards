function simmats=squareSimmats(simmats_ltv)
% converts set of row-vector simmats_ltv to square form (despite being
% rows, simmats are stacked along the 3rd dimension, just as square simmats
% would be. this avoids ambiguity when the simmats_ltv is square and could
% be either a single simmat or a number of vectorized simmats.)
% simmats may be bare or wrapped with meta-data in a struct object. they
% will be returned in the same format as passed.

if isstruct(simmats_ltv)
    % wrapped
    simmats_ltv_struct=simmats_ltv;
    simmats_ltv=unwrapSimmats(simmats_ltv_struct);
    
    nSimmats=size(simmats_ltv,3);
    simmats=[];
    for simmatI=1:nSimmats
        simmats=cat(3,simmats,squareSimmat(simmats_ltv(:,:,simmatI)));
    end
    
    simmats=wrapSimmats(simmats,simmats_ltv_struct);
else
    % bare
    nSimmats=size(simmats_ltv,3);
    simmats=[];
    for simmatI=1:nSimmats
        simmats=cat(3,simmats,squareSimmat(simmats_ltv(:,:,simmatI)));
    end
end
