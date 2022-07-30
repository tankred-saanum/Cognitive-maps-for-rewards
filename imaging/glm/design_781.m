% (c) Mona Garvert 2019 MPI CBS


clear all

% Prepare compilers
% cfg.verbose = 0;
% try cfg_getfile(cfg); end

addpath(genpath('/data/p_02071/spm'));

root            = '/data/pt_02071/choice-maps/imagingData/sub-';
datadir        	= '/data/pt_02071/choice-maps/preprocessed_data/fmriprep/sub-';
nblocks     	= 3;

subj        = XXsubjIDXX;
session     = XXsessionXX;
design_name = 'XXdesignXX';

% for subj = 101:130
%     for session = 2:3
%         try

% If not normalising then create own mask to use for epis- check this as we
% don't have an ideal struct to use as a mask. Must be big enough
% 
% disp(['%%%%%%%%% Starting model ',design_name,' for subject: ',num2str(subj), ' and session: ', num2str(session),' %%%%%%%%%']);
% 
% % savedir = '/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data';
% 
% % load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
% % arena=  squareform(data.arena_space{1}.distMatsForAllTrials_ltv);
% 
% % create design directory, specify units of design, TR and slices
% % ====================================================================== %
% 
% fmri_spec.dir             = {[root,num2str(subj),'/ses-',num2str(session),'/1stLevel/design_',design_name,'/']}; % directory for SPM.mat file
% fmri_spec.timing.units    = 'scans';   % units for design: seconds or scans?
% fmri_spec.timing.RT       = 2;    % TR
% fmri_spec.timing.fmri_t   = 60;       % microtime resolution (time-bins per scan) = slices
% fmri_spec.timing.fmri_t0  = 30;        % microtime onset = time bin at which regressors coincide with data acquisiton
% 
% if ~exist([fmri_spec.dir{1}],'dir'); mkdir([fmri_spec.dir{1}]); end
% 
% delete ([fmri_spec.dir{1},'SPM.mat']);
% 
% for run = 1:nblocks
%     
%     % number of scans and data
%     % ====================================================================== %
%     
%     
%     epi = [datadir,num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-object_run-0',num2str(run),'_space-MNI152NLin6Asym_desc-preproc_bold'];
%     nb = nifti([epi,'.nii']);
%     
%     for e = 1:nb.dat.dim(4)
%         fmri_spec.sess(run).scans{e,1} = [epi,'.nii',',',num2str(e)];
%     end
%     
%     fmri_spec.sess(run).nscan = length(fmri_spec.sess(run).scans);
%     
%     % task conditions - EVs of design matrix
%     % =================================================================== %
%     c = 0; % condition counter
%     
%     % Load events
%     events =  tdfread(['/data/p_02071/choice-maps/my_dataset/sub-',num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-object_run-0',num2str(run),'_events.tsv']);
%     load(['/data/p_02071/choice-maps/my_dataset/sub-',num2str(subj),'/sub-',num2str(subj),'_exploration_data.mat']);
% 
%     % Load distance matrices
%     matrixDir = '/data/pt_02071/choice-maps/tankred_modling/fmri1409/matrices/';
%     distance = load(fullfile(matrixDir,num2str(subj),'euclidean_kernel_matrix.csv'));
%     SR = load(fullfile(matrixDir,num2str(subj),'SR_kernel_matrix.csv'));
%     
%     %% repetition trials
%     c = c+1;
%     rep = find((events.object(2:end)-events.object(1:end-1)) == 0)+1;    
%     choicetrials = find(~strcmp(events.choicetype,"n/a"));
%     rep = rep(~ismember(rep,choicetrials));             % remove repeated objects
%             
%     fmri_spec.sess(run).cond(c).name                = 'repetition';
%     fmri_spec.sess(run).cond(c).onset               = events.onset(rep)/fmri_spec.timing.RT; %
%     fmri_spec.sess(run).cond(c).duration            = events.duration(rep)/fmri_spec.timing.RT; %constant
%     fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
%     fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
%     fmri_spec.sess(run).cond(c).orth                = 0;
%             
%     
%     sw = ["stay","switch"];
%     
%     RSA_events = find(strcmp(events.choicetype,"n/a"));
%     choice_events = find(~strcmp(events.choicetype,"n/a"));
%             
%     % Does an RSA trial immediately follow a choice? *ignore those
%     follow_choice = zeros(168,1);
%     follow_choice(choice_events+1) = 1;
%     follow_choice = follow_choice(1:168);
%     
%     %% Object onsets
%     for obj = 1:12
%         c = c+1;
%         
%         ix = find(events.object == obj & strcmp(events.choicetype,"n/a") & ~follow_choice);
%         ix = ix(~ismember(ix,rep));             % remove repeated objects
%         
%         ix(ix==1) = [];
%         exp_dist = []; SR_dist = [];
%         for i = 1:length(ix)
%             exp_dist(i) = distance(events.object(ix(i)),events.object(ix(i)-1));
%             SR_dist(i) = SR(events.object(ix(i)),events.object(ix(i)-1));
%         end
%         
%         fmri_spec.sess(run).cond(c).name                = ['objects_',num2str(obj)];
%         fmri_spec.sess(run).cond(c).onset               = events.onset(ix)/fmri_spec.timing.RT; %
%         fmri_spec.sess(run).cond(c).duration            = events.duration(ix)/fmri_spec.timing.RT; %constant
%         fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
%         fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
%         fmri_spec.sess(run).cond(c).orth                = 0;
%         
%         fmri_spec.sess(run).cond(c).pmod(1).param       = zscore(exp_dist,0); % vector containing parametric values, demeaned
%         fmri_spec.sess(run).cond(c).pmod(1).name        = 'experienced_distance'; % name to call parameters
%         fmri_spec.sess(run).cond(c).pmod(1).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
%         
%         fmri_spec.sess(run).cond(c).pmod(2).param       = zscore(SR_dist,0); % vector containing parametric values, demeaned
%         fmri_spec.sess(run).cond(c).pmod(2).name        = 'experienced_distance'; % name to call parameters
%         fmri_spec.sess(run).cond(c).pmod(2).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
%         
%     end
%     
%     
%     %% choice trials
%    
%     
%     c = c+1;
%     ix = ~strcmp(events.choicetype,"n/a") & events.duration > 0.2;  % remove really fast events
%     
%     fmri_spec.sess(run).cond(c).name                = 'choices';
%     fmri_spec.sess(run).cond(c).onset               = events.onset(ix)/fmri_spec.timing.RT;
%     fmri_spec.sess(run).cond(c).duration            = events.duration(ix)/fmri_spec.timing.RT; %constant
%     fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
%     fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
%     fmri_spec.sess(run).cond(c).orth                = 0;
%     
%     fmri_spec.sess(run).cond(c).pmod(1).param       = detrend(events.chosen_distance(ix),0); % vector containing parametric values
%     fmri_spec.sess(run).cond(c).pmod(1).name        = 'chosen distance'; % name to call parameters
%     fmri_spec.sess(run).cond(c).pmod(1).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
%     
%     fmri_spec.sess(run).cond(c).pmod(2).param       = detrend(events.unchosen_distance(ix),0); % vector containing parametric values
%     fmri_spec.sess(run).cond(c).pmod(2).name        = 'unchosen distance'; % name to call parameters
%     fmri_spec.sess(run).cond(c).pmod(2).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
%     
%     
%     %% button
%     c = c+1;
%     
%     fmri_spec.sess(run).cond(c).name                = 'button';
%     fmri_spec.sess(run).cond(c).onset               = events.button(~isnan(events.button) & events.duration > 0.2)/fmri_spec.timing.RT; %
%     fmri_spec.sess(run).cond(c).duration            = 0; %constant
%     fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
%     fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
%     
%     
%     % other regressors: motion regressors- once physio regressors have been
%     % appended call this file name
%     % ======================================================================= %
%     fmri_spec.sess(run).multi           = {''};
%     fmri_spec.sess(run).regress         = struct('name', {}, 'val', {});
%     
%     % Confound regressors as generated using create_confound_regressor
%     % script
%     fmri_spec.sess(run).multi_reg = {['/data/pt_02071/choice-maps/preprocessed_data/fmriprep/sub-',num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-object_run-0',num2str(run),'_rp.txt']};
%     
%     
%     
%     fmri_spec.sess(run).hpf             = 128;
%     fmri_spec.fact                      = struct('name', {}, 'levels', {});
%     fmri_spec.bases.hrf.derivs          = [0 0]; %The default is to have these set to zero. Set to one for temporal derivatives
%     fmri_spec.volt                      = 1;
%     fmri_spec.global                  	= 'None';
%     fmri_spec.mthresh                   = 0.000001;
%     fmri_spec.cvi                       = 'FAST';
%     fmri_spec.mask                      = {[datadir,num2str(subj),'/anat/sub-',num2str(subj),'_space-MNI152NLin6Asym_desc-brain_mask.nii']};
%     disp(run)
%     disp(c)
%     
% end
% 
% 
% % Run design specification
% % ======================================================================= %
% matlabbatch{1}.spm.stats.fmri_spec = fmri_spec;
% design_specification = matlabbatch;
% 
% % delete ([fmri_spec.dir{1},'/SPM.mat']);
% 
% save ([fmri_spec.dir{1},'/design_spec.mat'],'matlabbatch');
% disp(['%%%%%%%%% Starting design spec for session: ', num2str(session),'%%%%%%%%%']);
% spm_jobman('run',design_specification);
% 
% clear matlabbatch;
% 
% 
% % Estimate
% % ======================================================================= %
% matlabbatch{1}.spm.stats.fmri_est.spmmat = {[fmri_spec.dir{1},'/SPM.mat']};
% matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
% estimation = matlabbatch;
% save([fmri_spec.dir{1},'/design.mat'],'design_specification','estimation');
% 
% disp(['%%%%%%%%% Starting estimation for session: ', num2str(session),'%%%%%%%%%']);
% spm_jobman('run',estimation);

clear epiimgs fmri_spec R mot_reg;
clear matlabbatch; clear fmri_spec;

disp('%%%%%%%%% Computing contrast maps %%%%%%%%%')
cd('/data/p_02071/choice-maps/scripts/designs')
contrast_781(subj,session);
