function contrast_781(subj,session)
% clear all

% for subj = 101:152
% for session=2:3
%     try

root            = '/data/pt_02071/choice-maps/imagingData/sub-';
nblocks         = 3;

% Name of the experimental design. Start with 2* for RSA-designs and 1* for
% more basic designs


design_name     = '781';

factorial_design.dir = {[root,num2str(subj),'/ses-',num2str(session),'/1stLevel/design_',design_name]};
cd(factorial_design.dir{1})

load('SPM.mat');


% how many columns needed for which condition
% ====================================================================== %

col.sess        = nblocks;              % one constant per block
col.motion      = 0;     % motion only %% 6 motion regressors per session and 17 physio regressors
col.cond        = length(SPM.Sess(1).col)-col.motion;

col.general     = zeros(1,col.cond);  % condition only
col.total.all   = nblocks*(col.cond+col.motion) + col.sess;
col.block       = repmat([col.general zeros(1,col.motion)],col.sess,1);

% all columns together
% ====================================================================== %

disp(['Design matrix should have ',num2str(col.total.all),' columns']);

concounter = 1;

% % ADAPTATION
% ====================================================================== %
col.names = [];
counter = 0;
for i = 1:size(SPM.Sess(1).U,2)
    for j = 1:size(SPM.Sess(1).U(i).name,2)
        counter = counter + 1;
        col.names{counter} = [num2str(counter), '  -  ', SPM.Sess(1).U(i).name{j}];
        disp([num2str(counter), '  -  ', SPM.Sess(1).U(i).name{j}]);
    end
end




% 1. Object Onset
ci_pos = 2:3:25;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
stim_onset = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'allOnset'; %name
stats.con.consess{concounter}.tcon.convec    = stim_onset;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 2. button
ci_pos = 26;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
button = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'button'; %name
stats.con.consess{concounter}.tcon.convec    = button;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 3. Distance
ci_pos = 3:3:37;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
distance = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
ix = find(distance); varix = var(SPM.xX.X(:,ix));

% remove empty regressors
distance(ix(varix==0)) = 0;
stats.con.consess{concounter}.tcon.name      = 'allDistance'; %name
stats.con.consess{concounter}.tcon.convec    = distance;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;

% 4. SR
ci_pos = 4:3:37;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
sr = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
ix = find(sr); varix = var(SPM.xX.X(:,ix));

% remove empty regressors
sr(ix(varix==0)) = 0;
stats.con.consess{concounter}.tcon.name      = 'SRDistance'; %name
stats.con.consess{concounter}.tcon.convec    = sr;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;

% 5. eu and SR
ci_pos = [3:3:37 4:3:37];
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
sr = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
ix = find(sr); varix = var(SPM.xX.X(:,ix));

% remove empty regressors
sr(ix(varix==0)) = 0;
stats.con.consess{concounter}.tcon.name      = 'SRDistance'; %name
stats.con.consess{concounter}.tcon.convec    = sr;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 6. Repetition
ci_pos = 1;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
repetition = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'repetition'; %name
stats.con.consess{concounter}.tcon.convec    = repetition;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 7. eu versus SR
ci_pos = 3:3:37;
ci_neg = 4:3:37;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
    block(i,ci_neg) = -11;
end
sr = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
ix = find(sr); varix = var(SPM.xX.X(:,ix));

% remove empty regressors
sr(ix(varix==0)) = 0;
stats.con.consess{concounter}.tcon.name      = 'eu_vs_sr'; %name
stats.con.consess{concounter}.tcon.convec    = sr;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


%%
stats.con.spmmat                    = {[factorial_design.dir{1},'/SPM.mat']}; % where to find SPM.mat on which estimation has already run

% show in results window
% ====================================================================== %

stats.results.spmmat                = {[factorial_design.dir{1},'/SPM.mat']}; %which SPM.mat
stats.results.conspec.titlestr      = '';   %whith title
stats.results.conspec.contrasts     = 1;    %which contrast
stats.results.conspec.threshdesc    = 'none'; %FWD or none
stats.results.conspec.thresh        = 0.05; %threshold
stats.results.conspec.extent        = 0;    %cluster extend
stats.results.conspec.mask          = struct('contrasts', {}, 'thresh', {}, 'mtype', {});
stats.results.units                 = 1;    %datatype:volumetric
stats.con.delete                    = 1;    %delete existing contrasts
stats.results.print                 = true; %print in results window

clear matlabbatch
matlabbatch{1}.spm.stats = stats;
contrast = matlabbatch;
save([factorial_design.dir{1},'/contrast.mat'],'contrast');
save([factorial_design.dir{1},'/contrastmatrix.mat'],'contrast');

spm_jobman('run',matlabbatch);

clear matlabbatch;


% Smooth contrast images
kernelsize = '8';

clear matlabbatch
mkdir ([factorial_design.dir{1},'/smooth/warp'])

% specify scans for each session and voxel shift map generated in
% previous step (vdm*)
images  = spm_select('List','con*nii');
for epi = 1:size(images,1)
    matlabbatch{1}.spm.spatial.smooth.data{epi,1} = fullfile(factorial_design.dir{1},images(epi,:));
end

k = str2num(kernelsize);
matlabbatch{1}.spm.spatial.smooth.fwhm = [k k k];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = ['s',kernelsize,'_'];

spm_jobman('run',matlabbatch);

images  = spm_select('List','spmT*nii');
for epi = 1:size(images,1)
    matlabbatch{1}.spm.spatial.smooth.data{epi,1} = fullfile(factorial_design.dir{1},images(epi,:));
end
spm_jobman('run',matlabbatch);

images  = spm_select('List','s_*nii');
for i = 1:size(images,1)
    movefile(deblank(images(i,:)),['smooth/',deblank(images(i,:))])
end




spm_jobman('run',matlabbatch);
movefile(['s',kernelsize,'_*'],[factorial_design.dir{1},'/smooth/'])

cd /data/p_02071/choice-maps/scripts/designs
end

