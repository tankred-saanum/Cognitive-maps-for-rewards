% Figure 2
clear 
close all

bdir = '/data/p_02071/choice-maps/Cognitive-maps-for-rewards/';
addpath([bdir,'figures/helper_scripts'])

startSubj = 101;
endSubj = 152;
subjList = startSubj:endSubj;
subjList([21,36,37,38]) = []; % removed due to technical issues

% Analyse population data
savedir = [bdir,'/behavior/datafiles/merged_data'];

% plot map structure
load([savedir,'/subj_',num2str(startSubj),'/data_',num2str(startSubj),'.mat']);
inference_objects = data.mat{3}.data.options.inference_objects;

choice = readtable([bdir,'/behavior/choice_beh.csv']);

%%

figure
subj = [101 145 131];
cm = [186 156 209]/255;

for figurec = 1:3
    load([savedir,'/subj_',num2str(subj(figurec)),'/data_',num2str(subj(figurec)),'.mat']);
    load([bdir,'/behavior/datafiles/merged_data/subj_',num2str(subj(figurec)),'/sub-',num2str(subj(figurec)),'_exploration_data.mat'])
    subplot(1,3,figurec)
    plot(explorationPath(:,1),explorationPath(:,2),'color','k');
    hold on

    writematrix(explorationPath,sprintf('source_data/figure2/source_data_fig2a_%u.csv',figurec)) 
    
    % plot object locations
    p_obj = data.mat{2}.data.objPositions*15;
    scatter(p_obj(:,1),p_obj(:,3),50,cm,'filled')
    rectangle('Position',[-1,-1,2,2]*15,'Curvature',[1,1], 'FaceColor','none','EdgeColor','k')
    axis equal off
    prepImg    
end

%%
clear position
prepost = {'pre','post'};

error_post = nan(48,2,2,12);

for figurec = 1:length(subjList)
    subj = subjList(figurec);
    disp(subj)
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
    radius = data.mat{2}.data.radius;
    p_obj = data.mat{2}.data.objPositions*radius;

    for session = 2:3
        for p = 1:2
            for trial = 0:11
                d = eval(['data.viz.session_',num2str(session),'.',prepost{p},'.positionObject.run_0.trial_',num2str(trial)]);
                
                position(figurec,d.stim+1,session-1,p,:) = [d.position(end,1),d.position(end,3)];
                error_post(figurec,session-1,p,d.stim+1) = pdist([squeeze(position(figurec,d.stim+1,session-1,p,:))';[p_obj(d.stim+1,1),p_obj(d.stim+1,3)']])/(radius*2)*100;
                
            end
        end
    end
end

figure;
for session = 1:2
    subplot(1,2,session)
    boxplot(squeeze(nanmean(error_post(:,session,:,:),4)),'positions',[1 4],'widths',0.7, 'colors','k');
    hold on
    prepImg
    ylim([0 45])
    hold on
    scatter(repmat([2,3],48,1),squeeze(nanmean(error_post(:,session,:,:),4)),'MarkerFaceColor','k')
    set(gca,'XTickLabel',{'pre','post'})
    ylabel('Object positioning memory error (% arena size)')
    plot([ones(48,1)*(2) ones(48,1)*(3)]',(squeeze(nanmean(error_post(:,session,:,:),4)))','k') 
end

writematrix([subjList', squeeze(nanmean(error_post(:,1,1,:),4)),squeeze(nanmean(error_post(:,1,2,:),4)),...
    squeeze(nanmean(error_post(:,2,1,:),4)),squeeze(nanmean(error_post(:,2,2,:),4))],'source_data/figure2/source_data_fig2b.csv')

w = table(categorical([1 1 2 2].'), categorical([1 2 1 2].'), 'VariableNames', {'cond', 'time'}); % within-design
d = table(squeeze(nanmean(error_post(:,1,1,:),4)),squeeze(nanmean(error_post(:,1,2,:),4)),...
    squeeze(nanmean(error_post(:,2,1,:),4)),squeeze(nanmean(error_post(:,2,2,:),4)), 'VariableNames', {'c1_t0', 'c1_t1', 'c2_t0', 'c2_t1'});
rm = fitrm(d, 'c2_t1-c1_t0 ~ 1', 'WithinDesign', w);
ranova(rm, 'withinmodel', 'cond*time')




%%
%%
p_obj = data.mat{2}.data.objPositions;
true_dist       = zscore(pdist([p_obj(:,1),p_obj(:,3)]));
for subj = startSubj:endSubj
    disp(['Subject ', num2str(subj)])
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
    try
    cr_by_trial(subj-100,:) = data.mat{3}.data.choice.cr;
    end
    if isfield(data,'arena_space')
        if isfield(data.arena_space{1},'distMatsForAllTrials_ltv')
            arena       = zscore(data.arena_space{1}.distMatsForAllTrials_ltv);
            arena_true_error(subj-100) = sqrt(mean((arena-true_dist).^2));
        else
            arena_true_error(subj-100) = nan;
        end
    else
        arena_true_error(subj-100) = nan;
    end
end
% remove subjects with incomplete data due to technical issues
arena_true_error([21,36,37,38]) = []; 
cr_by_trial([21,36,37,38],:) = [];
cr_by_trial = cr_by_trial;

writematrix(cr_by_trial,'source_data/figure2/source_data_fig2c.csv')

figure('Renderer', 'painters', 'Position', [10 10 900 900])
subplot(2,2,1)
c = repmat({'r','b'},1,5);
for i = 1:10
    meancr(:,i) = (nanmean(cr_by_trial(:,(i-1)*10+1:i*10),2));
end
plot(1:10,meancr','Color',[0.5, 0.5, 0.5, 0.5])
hold on
for i = 1:10
    boxplot(nanmean(cr_by_trial(:,(i-1)*10+1:i*10),2)*100,'color',c{i},'positions',i,'widths',0.7);
end
    
ylim([0 100])
prepImg
xlabel('Block')
ylabel('Percent correct')
set(gca,'XTick',1:10,'XTickLabel',1:10)

%%
edge = unique(choice.edges);

for map = 1:2
for i = 1:length(edge)
    chosenSideMap(map,i,:) = (choice.value(choice.map == map & choice.edges == edge(i)));
end
end
color1 = {'r','b'};    
color = {'r','b'};

subplot(2,2,2)
for map = 1:2
    cs = squeeze(chosenSideMap(map,:,:)+1)/2;
    cs(:,21)= [];
    if map == 1 
        plot(90:-20:-90,cs','Color',[1, 0, 0, 0.1])
    else
        plot(90:-20:-90,cs','Color',[0, 0, 1, 0.1])
    end
    hold on
end
for map = 1:2
    cs = squeeze(chosenSideMap(map,:,:)+1)/2;
    cs(:,21)= [];
    hold on
    errorbar(90:-20:-90,nanmean(cs,2),nanstd((cs),0,2)/sqrt(48),color{map},'LineStyle','none','LineWidth',2)
    hold on
    plot([90:-20:-90],nanmean(cs,2),color{map},'LineWidth',2)
    prepImg
    ylim([0 1])
    xlabel('Value difference right versus left')
    ylabel('Probability of choosing the right option')
    plot([90:-20:-90],0.5*ones(10,1),'--k')
    scatter([90:-20:-90],nanmean(cs,2),40,color{map},'filled','MarkerEdgeColor',color{map},...
        'MarkerFaceColor',color{map})
    [h,p,ci,stats] = ttest(nanmean(cs(1:5,:)),nanmean(cs(6:10,:)))    
end

T = array2table([subjList' squeeze(chosenSideMap(1,:,:)+1)'/2  squeeze(chosenSideMap(2,:,:)+1)'/2]);
writetable(T,'source_data/figure2/source_data_fig2d.csv')

%%
load([bdir,'/figures/data/population_data.mat']);
value_inference = all_data.value_rating(:,[data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]);
value_inference([21,36,37,38],:) = [];
colors = [1 0 0; 1 0 0; 0 0 1; 0 0 1];
        
subplot(2,2,3)
pos_scatter = [2 3 6 7];
set(gcf,'renderer','Painters')
boxplot((value_inference),'colors',colors,'positions',[1 4 5 8],'widths',0.7);
hold on
for i = 1:4
    scatter(ones(48,1)*(pos_scatter(i)),value_inference(:,i),'filled','MarkerEdgeColor',colors(i,:),...
        'MarkerFaceColor',colors(i,:), 'MarkerFaceAlpha',.3)
    hold on
    if i == 1 || i == 3
        plot([ones(48,1)*(pos_scatter(i)) ones(48,1)*(pos_scatter(i+1))]',value_inference(:,i:i+1)','k')
    end
end

writematrix([subjList', value_inference],'source_data/figure2/source_data_fig2e.csv')

hold on
prepImg
title('Value rating of inference objects')
ylim([0 100])
xlim([0 9])

%%
inference_all = [nanmean(value_inference(:,1),2) nanmean(value_inference(:,3),2) nanmean(value_inference(:,2),2) nanmean(value_inference(:,4),2)];
inference_all(1,:) = [];
inference_performance = [nanmean(value_inference(:,[1 3]),2) nanmean(value_inference(:,[2 4]),2)];
w = table(categorical([1 1 2 2].'), categorical([1 2 1 2].'), 'VariableNames', {'object', 'context'}); % within-design
d = table(inference_all(:,1), inference_all(:,2),inference_all(:,3),inference_all(:,4),'VariableNames', {'c1_o0', 'c1_o1', 'c2_o0', 'c2_o1'});
rm = fitrm(d, 'c2_o1-c1_o0 ~ 1', 'WithinDesign', w);
ranova(rm, 'withinmodel', 'object*context')


% compute squared error for inference rating
real_value = [data.mat{3}.data.settings.value(1,:) data.mat{3}.data.settings.value(2,:)];
real = repmat(real_value([data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]),52,1);
rate = all_data.value_rating(:,[data.mat{3}.data.options.inference_objects(1,:) data.mat{3}.data.options.inference_objects(2,:)+12]);

inference_sqError_real_rate = sqrt(sum((rate-real).^2,2)/4);
inference_sqError_real_rate([21,36,37,38],:) = [];

subplot(2,2,4)
set(gcf,'renderer','Painters')
scatter(arena_true_error,...
    inference_sqError_real_rate,'filled','MarkerEdgeColor',[0 0.5 0.5],...
    'MarkerFaceColor',[0 0.7 0.7], 'MarkerFaceAlpha',.3), lsline
[r,p,rlo,rup] = corrcoef(arena_true_error', inference_sqError_real_rate,'rows','complete')

writematrix([subjList', arena_true_error', inference_sqError_real_rate],'source_data/figure2/source_data_fig2f.csv')

prepImg
xlabel('Map reproduction error')
ylabel('Inference error error')
title(sprintf('r = %.2f, p = %.3f',r,p))

% Robust fit:
[b,stats] = robustfit(inference_sqError_real_rate,arena_true_error)
