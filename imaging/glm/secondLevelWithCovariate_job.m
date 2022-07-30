clear all
close all

% design you want to re-run with covariate
des = '15781';
bdir = ['/data/pt_02071/choice-maps/imagingData/2ndLevel/design_',des,'/session_3'];
c = '10_spatial_c1-c2_s8';%'05_RPE_euc_vs_temp_at_feedback_s8';%'03_allDistance_s8'; %'05_RPE_euc_vs_temp_at_feedback_s8';3

type = 'spatial_effect';
remove_entry = false(48,1);

% Load correct behavioral parameters
%     predEffects = readtable(['/data/pt_02071/choice-maps/tankred_modling/final_modeling/effects_and_weights.csv']);

if strcmp(des,'10781') || strcmp(des,'12781')
    predEffects = readtable(['/data/pt_02071/choice-maps/tankred_modling/fmri26.4.2022-compositional_hparams_true_locations/effects_and_weights_review_individualfits_true_locations.csv']);
    predEffectsArray = table2array(predEffects(:,2:end));
elseif strcmp(des,'781') || strcmp(des,'11781')  || strcmp(des,'13781')  || strcmp(des,'15781')
    predEffects = readtable(['/data/pt_02071/choice-maps/tankred_modling/dataforMona2.2.22/effects_and_weights_review.csv']);
    predEffectsArray = table2array(predEffects(:,2:end));
end

if strcmp(type, 'posteriorEucEffects')
    predEffects = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/data17.5.2022/weightChangeCovariates.csv'));
    pe = table2array(predEffects(:,1));
    
    roi{1} = 'posteriorEucEffects';
    
elseif strcmp(type, 'rpeDiffEffects')
    predEffects = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/data17.5.2022/weightChangeCovariates.csv'));
    pe = table2array(predEffects(:,2));
    
    roi{1} = 'rpeDiffEffects';
 
elseif strcmp(type,'spatial_weight')
    
    pe = predEffectsArray(:,3);
    roi{1} = 'spatial_weight';
    
elseif strcmp(type,'spatial_effect')
    
    pe = predEffectsArray(:,2);
    roi{1} = 'spatial_effect';

elseif strcmp(type,'temporal_effect')

    pe = predEffectsArray(:,1);
    roi{1} = 'temporal_effect';
    
elseif strcmp(type,'percent_correct')
    savedir = ['/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data'];
    load([savedir,'/population_data.mat'])
    pe= all_data.overall_cr;
    pe([21,36,37,38]) = [];
    
    roi{1} = type;
    
elseif strcmp(type,'inference_error')    
    savedir = ['/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data'];
    load([savedir,'/population_data.mat'])
    
    load([savedir,'/subj_101/data_101.mat']);
    real_value = [data.mat{3}.data.settings.value(1,:) data.mat{3}.data.settings.value(2,:)];
    real = repmat(real_value([data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]),52,1);
    rate = all_data.value_rating(:,[data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]);

    pe = sqrt(sum((rate-real).^2,2)/4);
    pe([21,36,37,38]) = [];
    
% first subject does not have inference ratings
    remove_entry = isnan(pe);

    
    roi{1} = type;

elseif strcmp(type,'slopes')
    slopes_and_inflection = (readtable(['/data/pt_02071/choice-maps/tankred_modling/final_modeling/logistic_slopes.csv']));
    slopes = table2array(slopes_and_inflection(:,1));
    pe = slopes;
    roi{1}  = 'slopes';
    
elseif strcmp(type,'roi')
    % extract covariate
    basedir{1} = ['/data/pt_02071/choice-maps/imagingData/2ndLevel/design_529/'];
    con{1} = '06_weight_update_s8'; %'03_allDistance_s8';%'05_RPE_euc_vs_temp_at_feedback_s8'; %'03_diff_weights_euc_s8';%'03_value_s8';%'03_allDistance_s8';%
    roi{1}  = '529_3_06_s8_thr3p3_leftHippoc';%'177_diff_03_s8_thr2p41_leftOFC';%'179_diff_03_thr2p41_bilatEC';% '179_3_03_s8_thr3p3_bilatEC';%
    session = '3';
    
    % load data
    maxSubj = 152;
    for subj = 101:maxSubj
        if subj ~= 121 && subj ~= 136 && subj ~= 137 && subj ~= 138
            if exist(fullfile(basedir{1},'mask',roi{1},['session_',session],con{1},sprintf('%d_%s_%s_%s.txt',subj,session,roi{1},con{1})),'file')
                pe(subj-100) = load(fullfile(basedir{1},'mask',roi{1},['session_',session],con{1},sprintf('%d_%s_%s_%s.txt',subj,session,roi{1},con{1})));
            else
                pe(subj-100) = nan;
            end
        else
            pe(subj-100) = nan;
        end
    end
    
    pe([21,36,37,38]) = [];
    
elseif strcmp(type,'robustfit_predictability_diff_euc')
    % find correct index for this subject
    all_subj = 101:152;
    all_subj([21,36,37,38]) = [];

    % other
    trial_data = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/final_modeling/individual_trial_data_models.csv'),'ReadRowNames', true);
    
    for i = 1:48
        subj = all_subj(i);
        subjIX = find(all_subj==subj);
        trial_ID = (subjIX-1)*100+1:subjIX*100;
        
        weights_euc = table2array(trial_data(trial_ID,'weights_euc'));
        diff_euc = diff(weights_euc);
        
        
        RPE_euc(:,i) = table2array(trial_data(trial_ID,'RPE_euc'));
        RPE_temp(:,i) = table2array(trial_data(trial_ID,'RPE_temp'));
        predictability = abs(RPE_euc(1:99,i)) - abs(RPE_temp(1:99,i));
        
        [b(i,:)] = robustfit(predictability,diff_euc);
    end
   pe = b(:,2);
   roi{1} = 'robustfit_predictability_diff_euc';

elseif strcmp(type,'regress_predictability_diff_euc')
    % find correct index for this subject
    all_subj = 101:152;
    all_subj([21,36,37,38]) = [];

    % other
    trial_data = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/final_modeling/individual_trial_data_models.csv'),'ReadRowNames', true);
    
    for i = 1:48
        subj = all_subj(i);
        subjIX = find(all_subj==subj);
        trial_ID = (subjIX-1)*100+1:subjIX*100;
        
        weights_euc = table2array(trial_data(trial_ID,'weights_euc'));
        diff_euc = diff(weights_euc);
        
        
        RPE_euc(:,i) = table2array(trial_data(trial_ID,'RPE_euc'));
        RPE_temp(:,i) = table2array(trial_data(trial_ID,'RPE_temp'));
        predictability = abs(RPE_euc(1:99,i)) - abs(RPE_temp(1:99,i));
        
        [b(i,:)] = regress(diff_euc,[ones(99,1) predictability]);
    end
   pe = b(:,2);
   
   
   roi{1}  = 'regress_predictability_diff_euc';
elseif strcmp(type,'robustfit_predictability_weights_euc')
    % find correct index for this subject
    all_subj = 101:152;
    all_subj([21,36,37,38]) = [];

    % other
    trial_data = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/final_modeling/individual_trial_data_models.csv'),'ReadRowNames', true);
    
    for i = 1:48
        subj = all_subj(i);
        subjIX = find(all_subj==subj);
        trial_ID = (subjIX-1)*100+1:subjIX*100;
        
        weights_euc = table2array(trial_data(trial_ID,'weights_euc'));
        diff_euc = diff(weights_euc);
        
        
        RPE_euc(:,i) = table2array(trial_data(trial_ID,'RPE_euc'));
        RPE_temp(:,i) = table2array(trial_data(trial_ID,'RPE_temp'));
        predictability = abs(RPE_euc(1:99,i)) - abs(RPE_temp(1:99,i));
        
        [b(i,:)] = robustfit(predictability,weights_euc(2:100));
    end
   pe = b(:,2);
   roi{1}  = 'robustfit_predictability_weights_euc';

elseif strcmp(type,'regress_predictability_weights_euc')
    % find correct index for this subject
    all_subj = 101:152;
    all_subj([21,36,37,38]) = [];

    % other
    trial_data = readtable(fullfile('/data/pt_02071/choice-maps/tankred_modling/final_modeling/individual_trial_data_models.csv'),'ReadRowNames', true);
    
    for i = 1:48
        subj = all_subj(i);
        subjIX = find(all_subj==subj);
        trial_ID = (subjIX-1)*100+1:subjIX*100;
        
        weights_euc = table2array(trial_data(trial_ID,'weights_euc'));
        diff_euc = diff(weights_euc);
        
        
        RPE_euc(:,i) = table2array(trial_data(trial_ID,'RPE_euc'));
        RPE_temp(:,i) = table2array(trial_data(trial_ID,'RPE_temp'));
        predictability = abs(RPE_euc(1:99,i)) - abs(RPE_temp(1:99,i));
        
        [b(i,:)] = regress(weights_euc(2:100),[ones(99,1) predictability]);
    end
   pe = b(:,2);
   
   
   roi{1}  = 'regress_predictability_weights_euc';
end




pe_ix = 101:152;
pe_ix([21,36,37,38]) = [];

dir = fullfile(bdir,c,roi{1});
mkdir(dir)

%-----------------------------------------------------------------------
% Job saved on 13-Jul-2021 11:17:01 by cfg_util (rev $Rev: 6942 $)
% spm SPM - SPM12 (7219)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
clear whichsubj
matlabbatch{1}.spm.stats.factorial_design.dir = {dir};
delete([dir '/SPM.mat']);
cd(fullfile(bdir,c))
contrast_images  = spm_select('List',['*_s8_con*']);
for epi = 1:size(contrast_images,1)
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans{epi,1} = fullfile(bdir,c,contrast_images(epi,:));
    whichsubj(epi,:) = str2num(contrast_images(epi,1:3));
end
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans(remove_entry,:) = [];

pe = pe(ismember(pe_ix,whichsubj));
pe(remove_entry==1) = [];

%%
matlabbatch{1}.spm.stats.factorial_design.cov.c = detrend(pe,0);
matlabbatch{1}.spm.stats.factorial_design.cov.cname = roi{1};
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};          
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

spm_jobman('run',matlabbatch);
clear matlabbatch

matlabbatch{1}.spm.stats.fmri_est.spmmat = {[dir '/SPM.mat']};
matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
matlabbatch{2}.spm.stats.con.spmmat = {[dir '/SPM.mat']};
matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'ttest';
matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = [1 0];
matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{2}.spm.stats.con.consess{2}.tcon.name = ['ttest_cov_',roi{1}];
matlabbatch{2}.spm.stats.con.consess{2}.tcon.weights = [0 1];
matlabbatch{2}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{2}.spm.stats.con.consess{3}.tcon.name = ['ttest_neg_cov_',roi{1}];
matlabbatch{2}.spm.stats.con.consess{3}.tcon.weights = [0 -1];
matlabbatch{2}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
matlabbatch{2}.spm.stats.con.delete = 0;
spm_jobman('run',matlabbatch);

for i = 1:3
    copyfile([dir,'/spmT_000',num2str(i),'.nii'],...
        [dir,'/',des,'_spmT_',matlabbatch{2}.spm.stats.con.consess{i}.tcon.name,'.nii'])
end
