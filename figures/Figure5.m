%% Experimental script to recreate figure 5
% Requires CanlabCore toolbox https://github.com/canlab/CanlabCore

clear all
close all

% Define baseline directory
bdir = '/data/p_02071/choice-maps/paper';

% Add dependencies
addpath(genpath([bdir,'/analysis/helper_scripts']))

%% define ROIs

% load imaging data
roi{1} = '781_3_03_s8_thr3p3_rightHC';
con{1} = 'spatial';
session{1}  = 'diff';

roi{2} = '781_3_03_s8_thr3p3_rightHC';
con{2} = 'temporal';
session{2}  = 'diff';

% spatial representation with hippocampal spatial weight update as a
% covariate
con{3}      = 'spatial';
roi{3}      = '781_diff_03_s8_thr3p3_rightHippoc_529_3_06_s8_thr3p3_leftHippoc';
session{3}  = 'diff';

% OFC evidence integration with hippocampal spatial weight update as a covariate
con{4}  = 'unsigned_PE_diff';
roi{4}  ='517_3_05_s8_thr3p3_bilatOFC_529_3_06_s8_thr3p3_leftHippoc';
session{4}  = '3';

% hippocampal spatial weight update
con{5}  = '06_weight_update_s8';
roi{5}  ='529_3_06_s8_thr3p3_leftHippoc';
session{5}  = '3';

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning

subjList = 101:152; 
subjList(removeID) = []; % individuals to remove due to technical problems during scanning
pid = subjList';

counter = 0;
for subj=subjList
    counter = counter +1;
    
    % parameter estimates 
    pe_spatial(counter,1)              = load(fullfile(bdir,'data','mask',roi{1},['session_',session{1}],con{1},sprintf('%d_%s_%s_%s.txt',subj,session{1},roi{1},con{1})));    % roi, subj, block
    pe_temporal(counter,1)             = load(fullfile(bdir,'data','mask',roi{2},['session_',session{2}],con{2},sprintf('%d_%s_%s_%s.txt',subj,session{2},roi{2},con{2})));
    pe_spatial_hcWeight_cov(counter,1) = load(fullfile(bdir,'data','mask',roi{3},['session_',session{3}],con{3},sprintf('%d_%s_%s_%s.txt',subj,session{3},roi{3},con{3})));    % roi, subj, block
    pe_OFC_hcWeight_cov(counter,1)     = load(fullfile(bdir,'data','mask',roi{4},['session_',session{4}],con{4},sprintf('%d_%s_%s_%s.txt',subj,session{4},roi{4},con{4})));
    pe_hcWeight(counter,1)             = load(fullfile(bdir,'data','mask',roi{5},['session_',session{5}],con{5},sprintf('%d_%s_%s_%s.txt',subj,session{5},roi{5},con{5})));

end



% swap sign for OFC effect as this was coded the other way around in the
% GLM
pe_OFC_hcWeight_cov = -pe_OFC_hcWeight_cov;

% remove outliers
pe_OFC_hcWeight_cov(pe_OFC_hcWeight_cov > (nanmean(pe_OFC_hcWeight_cov) + 3* nanstd(pe_OFC_hcWeight_cov))) = nan;



%% behavior 

% effects and weights
% Load modeling data
predEffects = table2array(readtable([bdir,'/data/effects_and_weights.csv']));
temporal_effect = predEffects(:,2);
spatial_effect = predEffects(:,3);
trialw = predEffects(:,7);

% performance
% Load behavioral data 
load([bdir,'/data/population_data.mat'])
correct = all_data.overall_cr;
correct(removeID) = [];
inference_error = all_data.inference_error;
inference_error(removeID) = [];

% slopes
slopes_and_inflection = readtable([bdir,'/data/logistic_slopes.csv']);
slopes = table2array(slopes_and_inflection(:,1));

% perform stsats to compare spatial and temporal influences on choice
[h,p,a,stats] = ttest(spatial_effect, temporal_effect)
sum(spatial_effect < temporal_effect)


%%
figure('Renderer', 'painters', 'Position', [10 10 1500 400])
subplot(1,3,1)
scatter(slopes,inference_error,'filled'),lsline
[r,p,rlo,rup] = corrcoef(slopes,inference_error,'rows','complete');
xlabel('Slope')
ylabel('Inference error')
title(sprintf('r = %.2f, p = %.3f',r,p));
prepImg

T = table(pid,slopes,inference_error);
writetable(T,'source_data/figure5/source_data_fig5b.csv')  

% Robust fit:
[b,stats] = robustfit(slopes,inference_error)
disp([stats.dfe stats.t(2) stats.p(2)])

subplot(1,3,2)
scatter(slopes,pe_spatial,'filled'),lsline
[r,p,rlo,rup] = corrcoef(slopes,pe_spatial,'rows','complete')
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('Slope')
ylabel('Change in hippocampal spatial enhancement effect')
prepImg


T = table(pid,slopes,pe_spatial);
writetable(T,'source_data/figure5/source_data_fig5c.csv')  

% Robust fit:
[b,stats] = robustfit(slopes,pe_spatial)
disp([stats.dfe stats.t(2) stats.p(2)])



subplot(1,3,3)
scatter(pe_spatial,pe_temporal','filled'),lsline
[r,p,rlo,rup] = corrcoef(pe_spatial,pe_temporal,'rows','complete')
xlabel('Change in hippocampal spatial enhancement effect')
ylabel('Change in hippocampal temporal enhancement effect')
title(sprintf('r = %.2f, p = %.3f',r,p));
prepImg

T = table(pid,pe_spatial,pe_temporal);
writetable(T,'source_data/figure5/source_data_fig5d.csv')  


[b,stats] = robustfit(pe_spatial',pe_temporal')
disp([stats.dfe stats.t(2) stats.p(2)])

%% Mediation analysis
T = table(pid,pe_OFC_hcWeight_cov,pe_spatial_hcWeight_cov,pe_hcWeight);
writetable(T,'source_data/figure5/source_data_fig5i.csv')  

<<<<<<< HEAD
[paths, stats] = mediation(-pe_OFC_hcWeight_cov,pe_spatial_hcWeight_cov, pe_hcWeight, 'boot', 'plots', 'verbose', 'bootsamples', 10000);
=======
[paths, stats] = mediation(pe_OFC_hcWeight_cov,pe_spatial_hcWeight_cov, pe_hcWeight, 'boot', 'plots', 'verbose', 'bootsamples', 10000);
>>>>>>> 71ec57fbd4141d3afce91b23a031084a151806da

