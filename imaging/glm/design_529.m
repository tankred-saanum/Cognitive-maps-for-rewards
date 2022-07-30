%% (c) Mona Garvert 2019 MPI CBS
clear all

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

disp(['%%%%%%%%% Starting model ',design_name,' for subject: ',num2str(subj), ' and session: ', num2str(session),' %%%%%%%%%']);

savedir = '/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data';

% load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
% arena=  squareform(data.arena_space{1}.distMatsForAllTrials_ltv);
load([savedir,'/population_data.mat'])

% create design directory, specify units of design, TR and slices
% ====================================================================== %

fmri_spec.dir             = {[root,num2str(subj),'/ses-',num2str(session),'/1stLevel/design_',design_name,'/']}; % directory for SPM.mat file
fmri_spec.timing.units    = 'scans';   % units for design: seconds or scans?
fmri_spec.timing.RT       = 2;    % TR
fmri_spec.timing.fmri_t   = 60;       % microtime resolution (time-bins per scan) = slices
fmri_spec.timing.fmri_t0  = 30;        % microtime onset = time bin at which regressors coincide with data acquisiton

if ~exist([fmri_spec.dir{1}],'dir'); mkdir([fmri_spec.dir{1}]); end

delete ([fmri_spec.dir{1},'SPM.mat']);

for run = 1
    
    % number of scans and data
    % ====================================================================== %
    
    
    epi = [datadir,num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-choice_space-MNI152NLin6Asym_desc-preproc_bold'];
    nb = nifti([epi,'.nii']);
    
    for e = 1:nb.dat.dim(4)
        fmri_spec.sess(run).scans{e,1} = [epi,'.nii',',',num2str(e)];
    end
    
    fmri_spec.sess(run).nscan = length(fmri_spec.sess(run).scans);
    
    % task conditions - EVs of design matrix
    % =================================================================== %
    c = 0; % condition counter
    
    % Load events
    events =  tdfread(['/data/p_02071/choice-maps/my_dataset/sub-',num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-choice_events.tsv']);
    
               
    modeltype = 3;
    modelindex = [1 2 3 6 7];
    
    dist_measure = {'zero','obj','subj','exploreForwATMin','exploreBackATMin'};

    savedir = ['/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data'];
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj)]);
    load(['/data/p_02071/choice-maps/choice_data/population_data.mat']);
    
    Modeldatadir = '/data/pt_02071/choice-maps/tankred_modling/final_modeling/fmri/';
    chosen = readtable(fullfile(Modeldatadir,'predictions/comp_preds_chosen.csv'),'ReadRowNames', true);
    unchosen = readtable(fullfile(Modeldatadir,'predictions/comp_preds_unchosen.csv'),'ReadRowNames', true);
    RPE = readtable(fullfile(Modeldatadir,'predictions/comp_RPE.csv'),'ReadRowNames', true);
    
    cv = table2array(chosen(num2str(subj),:));
    uv = table2array(unchosen(num2str(subj),:));
    pe = table2array(RPE(num2str(subj),:));
    m  = data.mat{3}.data.choice.map;
    
    % other 
    trial_data = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/final_modeling/individual_trial_data_models.csv'),'ReadRowNames', true);

    % find correct index for this subject
    all_subj = 101:152;
    all_subj([21,36,37,38]) = [];
    subjIX = find(all_subj==subj);
    trial_ID = (subjIX-1)*100+1:subjIX*100;

    nll_euc = table2array(trial_data(trial_ID,'nll_euc'));
    nll_temp = table2array(trial_data(trial_ID,'nll_temp'));
    RPE_euc = table2array(trial_data(trial_ID,'RPE_euc'));
    RPE_temp = table2array(trial_data(trial_ID,'RPE_temp'));
    posterior_euc = table2array(trial_data(trial_ID,'posterior_euc'));
    pred_euc = table2array(trial_data(trial_ID,'pred_euc'));
    pred_temp = table2array(trial_data(trial_ID,'pred_temp'));
    delta_w = table2array(trial_data(trial_ID,'delta_w'));
    
    
    %% choice trials
    c = c+1;
    ix_onset = strcmp(events.trial_type,"choice") & events.duration > 0.2;  % remove really fast events
    ix_onset_val = (find(ix_onset)-1)/2+1;
    
    fmri_spec.sess(run).cond(c).name                = 'choices';
    fmri_spec.sess(run).cond(c).onset               = events.onset(ix_onset)/fmri_spec.timing.RT;
    fmri_spec.sess(run).cond(c).duration            = events.duration(ix_onset)/fmri_spec.timing.RT; %constant
    fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
    fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
    fmri_spec.sess(run).cond(c).orth                = 0;
    
    fmri_spec.sess(run).cond(c).pmod(1).param       = detrend(cv(ix_onset_val),0); % vector containing parametric values
    fmri_spec.sess(run).cond(c).pmod(1).name        = 'chosen value'; % name to call parameters
    fmri_spec.sess(run).cond(c).pmod(1).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
    
    fmri_spec.sess(run).cond(c).pmod(2).param       = detrend(uv(ix_onset_val),0); % vector containing parametric values
    fmri_spec.sess(run).cond(c).pmod(2).name        = 'unchosen value'; % name to call parameters
    fmri_spec.sess(run).cond(c).pmod(2).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)
    
    c = c+1;
    ix_feedback = strcmp(events.trial_type,"feedback") & events.duration > 0.2;  % remove really fast events
    ix_feedback_val = events.duration(2:2:end)  > 0.2;
    
    weights_euc = table2array(trial_data(trial_ID,'weights_euc'));
    weights_euc = weights_euc(ix_feedback_val);
    diff_euc = diff(weights_euc);
    
    fmri_spec.sess(run).cond(c).name                = 'feedback';
    fmri_spec.sess(run).cond(c).onset               = events.onset(ix_feedback)/fmri_spec.timing.RT;
    fmri_spec.sess(run).cond(c).duration            = events.duration(ix_feedback)/fmri_spec.timing.RT; %constant
    fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
    fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
    fmri_spec.sess(run).cond(c).orth                = 0;
    
    fmri_spec.sess(run).cond(c).pmod(1).param       = [0; detrend(diff_euc,0)]; % vector containing parametric values
    fmri_spec.sess(run).cond(c).pmod(1).name        = 'euc_weight_update'; % name to call parameters
    fmri_spec.sess(run).cond(c).pmod(1).poly        = 1; %First order polynomial expansion, always leave on 1 (so for all parametric regressors the same, could define in a loop)

    
    %% button
    c = c+1;
    
    fmri_spec.sess(run).cond(c).name                = 'button';
    fmri_spec.sess(run).cond(c).onset               = events.button(ix_feedback & events.duration > 0.2)/fmri_spec.timing.RT; %
    fmri_spec.sess(run).cond(c).duration            = 0; %constant
    fmri_spec.sess(run).cond(c).tmod                = 0;      % time modulation
    fmri_spec.sess(run).cond(c).pmod                = struct('name', {}, 'param', {}, 'poly', {});
    
    
    % other regressors: motion regressors- once physio regressors have been
    % appended call this file name
    % ======================================================================= %
    fmri_spec.sess(run).multi           = {''};
    fmri_spec.sess(run).regress         = struct('name', {}, 'val', {});
    
    % Confound regressors as generated using create_confound_regressor
    % script
    fmri_spec.sess(run).multi_reg = {['/data/pt_02071/choice-maps/preprocessed_data/fmriprep/sub-',num2str(subj),'/ses-',num2str(session),'/func/sub-',num2str(subj),'_ses-',num2str(session),'_task-choice_rp.txt']};
    
    
    
    fmri_spec.sess(run).hpf             = 128;
    fmri_spec.fact                      = struct('name', {}, 'levels', {});
    fmri_spec.bases.hrf.derivs          = [0 0]; %The default is to have these set to zero. Set to one for temporal derivatives
    fmri_spec.volt                      = 1;
    fmri_spec.global                  	= 'None';
    fmri_spec.mthresh                   = 0.000001;
    fmri_spec.cvi                       = 'FAST';
    fmri_spec.mask                      = {[datadir,num2str(subj),'/anat/sub-',num2str(subj),'_space-MNI152NLin6Asym_desc-brain_mask.nii']};
    disp(run)
    disp(c)
    
end


% Run design specification
% ======================================================================= %
matlabbatch{1}.spm.stats.fmri_spec = fmri_spec;
design_specification = matlabbatch;

delete ([fmri_spec.dir{1},'/SPM.mat']);

save ([fmri_spec.dir{1},'/design_spec.mat'],'matlabbatch');
disp(['%%%%%%%%% Starting design spec for session: ', num2str(session),'%%%%%%%%%']);
spm_jobman('run',design_specification);

clear matlabbatch;


% Estimate
% ======================================================================= %
matlabbatch{1}.spm.stats.fmri_est.spmmat = {[fmri_spec.dir{1},'/SPM.mat']};
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
estimation = matlabbatch;
save([fmri_spec.dir{1},'/design.mat'],'design_specification','estimation');

disp(['%%%%%%%%% Starting estimation for session: ', num2str(session),'%%%%%%%%%']);
spm_jobman('run',estimation);

clear epiimgs fmri_spec R mot_reg;
clear matlabbatch; clear fmri_spec;

disp('%%%%%%%%% Computing contrast maps %%%%%%%%%')
cd('/data/p_02071/choice-maps/scripts/designs')
contrast_529(subj,session);
%         catch
%         end
%     end
% end
