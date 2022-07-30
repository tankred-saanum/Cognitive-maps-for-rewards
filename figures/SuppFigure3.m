
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

removeID = [21, 36, 37, 38]; % individuals to remove due to technical problems during scanning

%% Modeled inference
load([bdir,'/figures/data/population_data.mat']);

tortuosity  = nan(52,2,2,12);
error       = nan(52,2,2,12);
pause       = nan(52,2,2,12);
rel_pause   = nan(52,2,2,12);
wayfinding_duration   = nan(52,2,2,12);
exploration = nan(52,1);

prepost = {'pre','post'};
for subj        = 101:152
    try
        disp(subj)
        
        load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
        
        % true object location
        true_position = data.mat{2}.data.objPositions;
        radius = data.mat{2}.data.radius;
          
        load(['/data/p_02071/choice-maps/my_dataset/sub-',num2str(subj),'/sub-',num2str(subj),'_exploration_data.mat'])
        exploration(subj-100) = explorationDuration;
        
        run = 0;
        for session = 2:3
            for p = 1:2
                
                if isfield(eval(['data.viz.session_', num2str(session),'.pre.positionObject']),['run_',num2str(run)])
                    if ~isempty(eval(['data.viz.session_', num2str(session),'.pre.positionObject.run_',num2str(run)]))
                        for trial = 0:11
                            
                            d = eval(['data.viz.session_',num2str(session),'.',prepost{p},'.positionObject.run_0.trial_',num2str(trial)]);
                            
                            expt.pos.session{session}.pre.positioning(d.stim+1,:) = d.position(end,[1,3]);%/(data.viz.radius*2);
                            
                            % Distance between start and end point
                            C = pdist([[d.position(1,1),d.position(1,3)];[d.position(end,1),d.position(end,3)]],'Euclidean');
                            
                            % cumulative length
                            L=0;
                            for i = 1:length(d.position)-1
                                L = L + pdist([[d.position(i,1),d.position(i,3)];[d.position(i+1,1),d.position(i+1,3)]],'Euclidean');
                            end
                            wayfinding_duration(subj-100,session-1,p,d.stim+1) = length(d.position);
                            
                            final_position = [d.position(end,1),d.position(end,3)];
                            error(subj-100,session-1,p,d.stim+1) = pdist([final_position;[true_position(d.stim+1,1)*radius,true_position(d.stim+1,3)*radius']]);
                            
                            pause(subj-100,session-1,p,d.stim+1) = sum(diff(d.position(:,1))==0 & diff(d.position(:,3))==0);
                            rel_pause(subj-100,session-1,p,d.stim+1) = sum(diff(d.position(:,1))==0 & diff(d.position(:,3))==0)/length(d.position);
                            
                            tortuosity(subj-100,session-1,p,d.stim+1) = C/L;
                            
                        end
                    end
                end
            end            
        end
    catch
    end
end

error(removeID,:,:,:)      = [];
tortuosity(removeID,:,:,:) = [];
pause(removeID,:,:,:)      = [];
rel_pause(removeID,:,:,:)  = [];
wayfinding_duration(removeID,:,:,:)  = [];
exploration(removeID)      = [];

% error(10,:,:,:) = nan;

measure{1} = exploration;
measure{2} = nanmean(nanmean(nanmean(error(:,:,:,:),4),3),2);
measure{3} = nanmean(nanmean(nanmean(wayfinding_duration(:,:,:,:),4),3),2);
measure{4} = nanmean(nanmean(nanmean(rel_pause(:,:,:,:),4),3),2);
measure{5} = nanmean(nanmean(nanmean(tortuosity(:,:,:,:),4),3),2);


[xpc, rpc] = pca([measure{1} measure{2} measure{3} measure{4} measure{5}]);
measure{6} = rpc(:,3);
measure{7} = rpc(:,2);

label{1} = 'exploration';
label{2} = 'error';
label{3} = 'wayfinding duration';
label{4} = 'pausing';
label{5} = 'tortuosity';


%%
% Is there any influence on the object positioning after the value
% learning?

clear position
prepost = {'pre','post'};
error_post = nan(48,2,2,12);
subjList = 101:152;
subjList(removeID) = [];

for c = 1:length(subjList)
    subj = subjList(c);
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
    radius = data.mat{2}.data.radius;
    p_obj = data.mat{2}.data.objPositions*radius;

    for session = 2:3
        for p = 1:2
            for trial = 0:11
                d = eval(['data.viz.session_',num2str(session),'.',prepost{p},'.positionObject.run_0.trial_',num2str(trial)]);
                
                position(c,d.stim+1,session-1,p,:) = [d.position(end,1),d.position(end,3)];
                error_post(c,session-1,p,d.stim+1) = pdist([squeeze(position(c,d.stim+1,session-1,p,:))';[p_obj(d.stim+1,1),p_obj(d.stim+1,3)']])/(radius*2)*100;
                
            end
        end
    end
end

%% Change over time
clear measure_complete
measure_complete{1} = (nanmean(error(:,:,:,:),4));
measure_complete{2} = (nanmean(rel_pause(:,:,:,:),4));
label{1} = 'Error';
label{2} = 'Pause';
colors = [1 0 0; 0 0 1];

figure('Renderer', 'painters', 'Position', [10 10 900 400])
for ix = 1:size(measure_complete,2)
    counter = 0;
    for i = 1:2
        for j = 1:2
            counter = counter + 1;
            m(:,counter) = measure_complete{ix}(:,i,j);
        end
    end

    subplot(1,3,ix);
    
    boxplot(m,'colors',colors,'positions',[1 4 6 9],'widths',0.7);
    hold on
    prepImg
    tos=repmat([2,3 7 8],48,1);
    scatter(tos(:),m(:))
    set(gca,'XTickLabel',{'pre','post'})
    ylabel(label{ix})
    hold on 

    if ix == 2
        ylim([0 1])
    end
end

%%
% slopes
slopes_and_inflection = (readtable([bdir,'/src/logistic_slopes.csv']));
slopes = table2array(slopes_and_inflection(:,1));


real_value   = all_data.real_value;
value_rating = all_data.value_rating;
value_rating(removeID,:) = []; % individuals to remove due to technical problems during scanning

for subj = 1:length(value_rating)
    try
        RMSE(subj) = sqrt(mean((real_value - value_rating(subj,:)).^2)); 
    end
end

% correlate RMSE with slopes
subplot(1,3,3)
scatter(RMSE, slopes,'filled')
lsline
[r,p] = corr(RMSE', slopes, 'rows','complete','type','Spearman');
title(sprintf('r = %.2f, p = %.3f',r,p))
xlabel('Value rating root mean square error')
ylabel('Slope')
prepImg
