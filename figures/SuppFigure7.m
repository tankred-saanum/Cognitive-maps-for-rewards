% (c) Mona Garvert, MPI CBS, June 2019

clear all
close all

% Define baseline directory
bdir = '//data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

% Add dependencies
addpath(genpath([bdir,'/figures/helper_scripts']))


% Analyse population data
savedir = [bdir,'/behavior/datafiles/merged_data'];

% plot map structure
startSubj = 101;
load([savedir,'/subj_',num2str(startSubj),'/data_',num2str(startSubj),'.mat']);
inference_objects = data.mat{3}.data.options.inference_objects;

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning

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
        
        load([bdir,'/behavior/datafiles/merged_data/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
        
        % true object location
        true_position = data.mat{2}.data.objPositions;
        radius = data.mat{2}.data.radius;
          
        load([bdir,'/behavior/datafiles/merged_data/subj_',num2str(subj),'/sub-',num2str(subj),'_exploration_data.mat'])
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
                            
                            tortuosity(subj-100,session-1,p,d.stim+1) = L/C;
                            
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

error(10,:,:,:) = nan;

measure{1} = exploration;
measure{2} = nanmean(nanmean(nanmean(error(:,:,:,:),4),3),2);
measure{3} = nanmean(nanmean(nanmean(wayfinding_duration(:,:,:,:),4),3),2);
measure{4} = nanmean(nanmean(nanmean(rel_pause(:,:,:,:),4),3),2);
measure{5} = nanmean(nanmean(nanmean(tortuosity(:,:,:,:),4),3),2);


[xpc, rpc] = pca([measure{1} measure{2} measure{3} measure{4} measure{5}]);
label{1} = 'Exploration duration';
label{2} = 'Replacement error';
label{3} = 'Wayfinding duration';
label{4} = 'Pausing';
label{5} = 'Tortuosity';

[m,p] = corr([measure{1} measure{2} measure{3} measure{4} measure{5}]);



%%

counter = 0;
figure('Renderer', 'painters', 'Position', [10 10 1200 1200])
for i = 1:size(measure,2)
    for j = 1:size(measure,2)
        counter = counter+1;
        if i > j
            subplot(size(measure,2),size(measure,2),counter)
            
            scatter(measure{i}, measure{j},'filled'), lsline
            [r,p] = corrcoef(measure{i}, measure{j},'rows','complete');
            title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)))
            xlabel(label{i})
            ylabel(label{j})
            prepImg
        end
    end
end


%% Load imaging data

% load imaging data
roi = '781_3_03_s8_thr3p3_rightHC';
con{1} = 'spatial';
con{2} = 'temporal';

session = '3';
subjIX = 101:152;
subjIX(removeID) = []; % individuals to remove due to technical problems during scanning

counter = 0;
for subj=subjIX
    counter = counter +1;
    pe_spatial(counter) = load(fullfile(bdir,'figures','masks',roi,['session_',session],'03_allDistance_s8',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'03_allDistance_s8')));
    pe_temporal(counter) = load(fullfile(bdir,'figures','masks',roi,['session_',session],'04_SRDistance_s8',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'04_SRDistance_s8')));
end

%%
figure; 
subplot(1,2,1)
i = 2;
scatter(measure{i}, pe_spatial,'k','filled'), lsline
[r,p] = corrcoef(measure{i}, pe_spatial,'rows','complete');
title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)))
xlabel(label{i})
ylabel('Spatial fMRI effect')
prepImg
subplot(1,2,2)
scatter(measure{i}, pe_temporal,'k','filled'), lsline
[r,p] = corrcoef(measure{i}, pe_temporal,'rows','complete');
title(sprintf('r = %.2f, p = %.3f',r(1,2),p(1,2)))
xlabel(label{i})
ylabel('Ppredictive fMRI effect')

prepImg
