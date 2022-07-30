%% Experimental script to recreate figure 4

clear all
close all

% Define baseline directory
bdir = '/data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

% Add dependencies
addpath(genpath([bdir,'/analysis/helper_scripts']))

% load imaging data
roi = '781_3_03_s8_thr3p3_rightHC';
con{1} = 'spatial';
con{2} = 'temporal';

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning

session = '3';
subjIX = 101:152;
subjIX(removeID) = []; % individuals to remove due to technical problems during scanning

counter = 0;
for subj=subjIX
    counter = counter +1;
    pe_spatial(counter) = load(fullfile(bdir,'figures','data','mask',roi,['session_',session],'spatial',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'spatial')));
    pe_temporal(counter) = load(fullfile(bdir,'figures','data','mask',roi,['session_',session],'temporal',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'temporal')));
end


% Load behavioral data 
load([bdir,'/figures/data/population_data.mat']);
inference_error = all_data.inference_error;
inference_error(removeID) = [];

% Load modeling data
predEffects = table2array(readtable([bdir,'/figures/data/effects_and_weights.csv']));
temporal_effect = predEffects(:,2);
spatial_effect = predEffects(:,3);

%% Test for normality
lillietest(pe_spatial)
lillietest(pe_temporal)
lillietest(spatial_effect)
lillietest(temporal_effect)
lillietest(inference_error)

%%
figure

subplot(2,2,1);
scatter(pe_spatial,spatial_effect,'filled')
lsline
[r,p] = corr(pe_spatial', spatial_effect,'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('Spatial fMRI effect')
ylabel('Spatial effect')
prepImg


% Robust fit:
[b,stats] = robustfit(pe_spatial', spatial_effect);
disp([stats.dfe stats.t(2) stats.p(2)])


subplot(2,2,2);
scatter(pe_spatial,inference_error,'filled'), lsline
[r,p] = corr(pe_spatial', inference_error,'rows','complete','type','Pearson');
xlabel('Spatial fMRI effect')
ylabel('Inference error'),
title(sprintf('r = %.2f, p = %.3f',r,p))
prepImg


subplot(2,2,3);
scatter(pe_temporal,temporal_effect,'filled')
lsline
[r,p] = corr(pe_temporal', temporal_effect,'rows','complete','type','Spearman');
title(sprintf('r = %.2f, p = %.3f',r,p));
ylabel('Temporal effect')
xlabel('Temporal fMRI effect')
prepImg

subplot(2,2,4);
scatter(pe_temporal,inference_error,'filled'), lsline
[r,p] = corr(pe_temporal', inference_error,'rows','complete','type','Pearson');
ylabel('Inference error'),
xlabel('Temporal fMRI effect')
title(sprintf('r = %.2f, p = %.3f',r,p))
prepImg



%% Mediation analysis
[paths, stats] = mediation(pe_spatial', inference_error, spatial_effect, 'boot', 'plots', 'verbose', 'bootsamples', 10000)


%% Sup
% Load distance matrices
clear dSRcorr
matrixDir = '/data/pt_02071/choice-maps/tankred_modling/fmri1409/matrices/';
for subj = 101:152
    try
    distance = load(fullfile(matrixDir,num2str(subj),'euclidean_kernel_matrix.csv')); 
    distance(eye(12)==1) = 0; 
    distance = squareform(distance);
    SR = load(fullfile(matrixDir,num2str(subj),'SR_kernel_matrix.csv'));
    SR(eye(12)==1) = 0; 
    SR = squareform(SR);
    dSRcorr(subj-100) = corr(distance',SR');
    end
end
dSRcorr([21,36,37,38]) = [];
    
%%
%% Test for normality
lillietest(dSRcorr)
lillietest(correct)
lillietest(spatial_effect)
lillietest(temporal_effect)
lillietest(inference_error)


figure; 
subplot(2,2,1)
scatter(dSRcorr,correct,'filled'), lsline
[r,p] = corr(dSRcorr',correct,'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('r spatial and temporal')
ylabel('Percent correct')
prepImg

subplot(2,2,2)
scatter(dSRcorr,inference_error,'filled'), lsline
[r,p] = corr(dSRcorr',inference_error,'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('r spatial and temporal')
ylabel('Inference error')
prepImg

subplot(2,2,3)
i=2;
scatter(dSRcorr,predEffectsArray(:,i),'filled'), lsline
[r,p] = corr(dSRcorr',predEffectsArray(:,i),'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('r spatial and temporal')
ylabel(predEffectsNames{i},'Interpreter','None')
prepImg

subplot(2,2,4)
i=1;
scatter(dSRcorr,predEffectsArray(:,i),'filled'), lsline
[r,p] = corr(dSRcorr',predEffectsArray(:,i),'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('r spatial and temporal')
ylabel(predEffectsNames{i},'Interpreter','None')
prepImg

lab = {'Spatial fMRI effect','Temporal fMRI effect'};
predEffectsNames{3} = 'Spatial weights';
sess = 2;
peRed = squeeze(pe(:,:,sess));
peRed(:,[21,36,37,38]) = [];

%%
figure
for c = 1:2
    subplot(2,2,c)
    scatter(dSRcorr',peRed(c,:),'filled')
    lsline
    [r,p] = corr(dSRcorr',peRed(c,:)', 'rows','complete','type','Pearson');
    title(sprintf('r = %.2f, p = %.2f',r,p));
    ylabel(lab{c})
    xlabel('r spatial and temporal')
    
end
prepImg

%% exploration path length
subj = 101:152;
subj([21,36,37,38]) = [];
for c=1:length(subj)
    
    load(['/data/p_02071/choice-maps/my_dataset/sub-',num2str(subj(c)),'/sub-',num2str(subj(c)),'_exploration_data.mat'])
    pathlength(c) = length(explorationPath);

end

%%
figure; 
subplot(2,2,1)
scatter(pathlength,correct,'filled'), lsline
[r,p] = corr(pathlength',correct,'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('pathlength')
ylabel('Percent correct')
prepImg

subplot(2,2,2)
scatter(pathlength,inference_error,'filled'), lsline
[r,p] = corr(pathlength',inference_error,'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('pathlength')
ylabel('Inference error')
prepImg

subplot(2,2,3)
i=2;
scatter(pathlength,predEffectsArray(:,i),'filled'), lsline
[r,p] = corr(pathlength',predEffectsArray(:,i),'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('pathlength')
ylabel(predEffectsNames{i},'Interpreter','None')
prepImg

subplot(2,2,4)
i=1;
scatter(pathlength,predEffectsArray(:,i),'filled'), lsline
[r,p] = corr(pathlength',predEffectsArray(:,i),'rows','complete','type','Pearson');
title(sprintf('r = %.2f, p = %.3f',r,p));
xlabel('pathlength')
ylabel(predEffectsNames{i},'Interpreter','None')
prepImg

lab = {'Spatial fMRI effect','Temporal fMRI effect'};
predEffectsNames{3} = 'Spatial weights';
sess = 2;
peRed = squeeze(pe(:,:,sess));
peRed(:,[21,36,37,38]) = [];

%%
figure
for c = 1:2
    subplot(2,2,c)
    scatter(pathlength',peRed(c,:),'filled')
    lsline
    [r,p] = corr(pathlength',peRed(c,:)', 'rows','complete','type','Pearson');
    title(sprintf('r = %.2f, p = %.2f',r,p));
    ylabel(lab{c})
    xlabel('pathlength')
    
end
prepImg
