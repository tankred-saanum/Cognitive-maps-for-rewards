%% Experimental script to recreate figure 4

clear all
close all

% Define baseline directory
bdir = '//data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

% Add dependencies
addpath(genpath([bdir,'/analysis/helper_scripts']))


% Analyse population data
savedir = [bdir,'/behavior/datafiles/merged_data'];

% plot map structure
startSubj = 101;
load([savedir,'/subj_',num2str(startSubj),'/data_',num2str(startSubj),'.mat']);
inference_objects = data.mat{3}.data.options.inference_objects;

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning

%% Modeled inference
load([bdir,'/figures/data/population_data.mat']);
inference_error = all_data.inference_error;
inference_error(removeID) = [];
       

values = readtable([bdir,'/src/fmri/predictions/comp_final_predictions1.csv']);
values1 = table2array(values); 

values = readtable([bdir,'/src/fmri/predictions/comp_final_predictions2.csv']);
values2 = table2array(values);

values = [values1(:,2:end) values2(:,2:end)];
model_inference = values(2:end,[data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]);



h = figure;
subplot(2,2,1)
pos_scatter = [2 3 6 7];
set(gcf,'renderer','Painters')
colors = [1 0 0; 1 0 0; 0 0 1; 0 0 1];
boxplot((model_inference),'colors',colors,'positions',[1 4 5 8],'widths',0.7);
hold on
for i = 1:4
    scatter(ones(48,1)*(pos_scatter(i)),model_inference(:,i),'filled','MarkerEdgeColor',colors(i,:),...
        'MarkerFaceColor',colors(i,:), 'MarkerFaceAlpha',.3)
    hold on
    if i == 1 || i == 3
        plot([ones(48,1)*(pos_scatter(i)) ones(48,1)*(pos_scatter(i+1))]',model_inference(:,i:i+1)','k')
    end
end

hold on
prepImg
title('Value rating of inference objects')
ylim([0 100])

%
real_value = [data.mat{3}.data.settings.value(1,:) data.mat{3}.data.settings.value(2,:)];
real = repmat(real_value([data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]),48,1);
rate = all_data.value_rating(:,[data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]);
rate([21,36,37,38],:) = [];

inference_sqError_real_rate = sqrt(sum((rate-real).^2,2)/4);
inference_sqError_real_model = sqrt(sum((rate-model_inference).^2,2)/4);

subplot(2,2,2)
scatter(inference_sqError_real_model,inference_sqError_real_rate,'filled'), lsline
[r,p] = corrcoef(inference_sqError_real_model,inference_sqError_real_rate,'rows','complete');
title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)));
xlabel('Model-predicted inference error')
ylabel('Inference error');
prepImg
kstest(inference_sqError_real_model)

% Robust fit:
[b,stats] = robustfit(inference_sqError_real_model,inference_sqError_real_rate);
disp([stats.dfe stats.t(2) stats.p(2)])


subplot(2,2,3)
% load data
predEffects = readtable(['/data/pt_02071/choice-maps/tankred_modling/final_modeling/effects_and_weights.csv']);
predEffectsArray = table2array(predEffects(:,2:end));
scatter(predEffectsArray(:,2),predEffectsArray(:,1),'filled'), lsline
[r,p] = corrcoef(predEffectsArray(:,2),predEffectsArray(:,1),'rows','complete');
title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)));
xlabel('Spatial effect')
ylabel('Temporal effect');
prepImg

% Robust fit:
[b,stats] = robustfit(predEffectsArray(:,2),predEffectsArray(:,1));
disp([stats.dfe stats.t(2) stats.p(2)])

subplot(2,2,4)
scatter(predEffectsArray(:,3),inference_sqError_real_rate,'filled'), lsline
[r,p] = corrcoef(predEffectsArray(:,3),inference_sqError_real_rate,'rows','complete');
title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)));
xlabel('Spatial weight')
ylabel('Inference error');
prepImg

% Robust fit:
[b,stats] = robustfit(predEffectsArray(:,3),inference_sqError_real_rate);
disp([stats.dfe stats.t(2) stats.p(2)])

%% 