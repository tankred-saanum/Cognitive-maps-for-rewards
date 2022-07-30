function contrast_529(subj,session)

root            = '/data/pt_02071/choice-maps/imagingData/sub-';
nblocks         = 1;

% Name of the experimental design. Start with 2* for RSA-designs and 1* for
% more basic designs


design_name     = '529';

factorial_design.dir = {[root,num2str(subj),'/ses-',num2str(session),'/1stLevel/design_',design_name]};
cd(factorial_design.dir{1})

load('SPM.mat');


% how many columns needed for which condition
% ====================================================================== %

col.sess        = nblocks;              % one constant per block
col.motion      = size(SPM.Sess(1).C.C,2);     % motion only %% 6 motion regressors per session and 17 physio regressors
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




% 1. Object Onset
ci_pos = 1;
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
ci_pos = 6;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
button = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'button'; %name
stats.con.consess{concounter}.tcon.convec    = button;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 3. chosen - unchosen
ci_pos = 2;
ci_neg = 3;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
    block(i,ci_neg) = -1;
end
chosen_unchosen = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'chosen-unchosen'; %name
stats.con.consess{concounter}.tcon.convec    = chosen_unchosen;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 4. chosen + unchosen
ci_pos = [2 3];
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
chosen_and_unchosen = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'chosen+unchosen'; %name
stats.con.consess{concounter}.tcon.convec    = chosen_and_unchosen;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 5. feedback
ci_pos = 4;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
feedback = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'feedback'; %name
stats.con.consess{concounter}.tcon.convec    = feedback;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 6. weight_update
ci_pos = 5;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
pe = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'weight_update'; %name
stats.con.consess{concounter}.tcon.convec    = pe;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 7. chosen
ci_pos = 2;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
chosen = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'chosen'; %name
stats.con.consess{concounter}.tcon.convec    = chosen;                   %contrast vector
stats.con.consess{concounter}.tcon.sessrep   = 'none';                   %replicate vector over sessions
concounter = concounter+1;


% 8. unchosen
ci_pos = 3;
block = col.block;
for i = 1:nblocks
    block(i,ci_pos) = 1;
end
unchosen = [reshape(block',1,length(block)*nblocks) zeros(1,nblocks)];
stats.con.consess{concounter}.tcon.name      = 'unchosen'; %name
stats.con.consess{concounter}.tcon.convec    = unchosen;                   %contrast vector
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

matlabbatch{1}.spm.stats = stats;
contrast = matlabbatch;
save([factorial_design.dir{1},'/contrast.mat'],'contrast');
save([factorial_design.dir{1},'/contrastmatrix.mat'],'contrast');

spm_jobman('run',matlabbatch);

clear matlabbatch;


%% Smooth contrast images

clear matlabbatch
mkdir ([factorial_design.dir{1},'/smooth/warp'])

% specify scans for each session and voxel shift map generated in
% previous step (vdm*)
images  = spm_select('List','con*nii');
for epi = 1:size(images,1)
    matlabbatch{1}.spm.spatial.smooth.data{epi,1} = fullfile(factorial_design.dir{1},images(epi,:));
end

matlabbatch{1}.spm.spatial.smooth.fwhm = [8 8 8];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = 's8_';

spm_jobman('run',matlabbatch);
% 
% images  = spm_select('List','spmT*nii');
% for epi = 1:size(images,1)
%     matlabbatch{1}.spm.spatial.smooth.data{epi,1} = fullfile(factorial_design.dir{1},images(epi,:));
% end
% spm_jobman('run',matlabbatch);
% 
% images  = spm_select('List','s_*nii');
% for i = 1:length(images)
%     movefile(deblank(images(i,:)),['smooth/',deblank(images(i,:))])
% end
% 
% 
% 
% 
spm_jobman('run',matlabbatch);
movefile('s8_*',[factorial_design.dir{1},'/smooth/'])
delete([factorial_design.dir{1},'/smooth/s_s_*'])
% 

cd /data/p_02071/choice-maps/scripts/designs