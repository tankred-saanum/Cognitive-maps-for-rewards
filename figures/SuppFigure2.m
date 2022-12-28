clear
close all

bdir = '/data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

addpath (genpath([bdir,'/figures/helper_scripts']))
savedir = [bdir,'behavior/datafiles/merged_data'];

load([bdir,'/figures/data/population_data.mat'])

subjList = 101:152;
subjList([21,36,37,38]) = [];

%%
clear position
prepost = {'pre','post'};

error_post = nan(48,2,2,12);

for c = 1:length(subjList)
    subj = subjList(c);
    disp(subj)
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

[h,p,tint,stats] = ttest(squeeze(nanmean(error_post(:,1,1,:),4)),squeeze(nanmean(error_post(:,1,2,:),4)))
[h,p,tint,stats] = ttest(squeeze(nanmean(error_post(:,2,1,:),4)),squeeze(nanmean(error_post(:,2,2,:),4)))

pos_error_allSess= ([squeeze(nanmean(error_post(:,1,:,:),4)) squeeze(nanmean(error_post(:,2,:,:),4)) ])




%%
   
% Plot all estimates
figure
set(gcf,'renderer','Painters')
for s = 1:12
    subplot(3,4,s)
    
    % convolve with Gaussian kernel
    scatter(position(:,s,1),position(:,s,2),'k','filled')
    hold on
    rectangle('Position',[-1,-1,2,2]*radius,'Curvature',[1,1], 'FaceColor','none','EdgeColor','k')
    
    scatter(p_obj(s,1),p_obj(s,3),'*y')
    
    axis equal off
    
end
set(gcf,'color','w')

%%
clear position
prepost = {'pre','post'};

error_post = nan(48,2,2,12);

for c = 1:length(subjList)
    subj = subjList(c);
    disp(subj)
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

%%
io = data.mat{3}.data.options.inference_objects;

value_rating = all_data.value_rating;
value_rating([21,36,37,38],:) = [];

allSubjValue = [nanmean(value_rating(:,1:12)); nanmean(value_rating(:,1:12))];
allSubjValue(1,io(1,:)) = nan;
allSubjValue(2,io(2,:)) = nan;
allSubjObject_value = nanmean(allSubjValue);
figure;
for session = 1:2
    subplot(1,2,session)
    boxplot((squeeze(beta(:,session,:,2))),'colors',colors,'positions',[1 4],'widths',0.7);
    hold on
    [h,p,tint,stats] = ttest(squeeze((beta(:,session,:,2))))
    prepImg
    ylim([-0.4 0.4])
    hold on
    scatter(repmat([2,3],48,1),squeeze(beta(:,session,:,2)),'MarkerFaceColor',colors(session,:))
    set(gca,'XTickLabel',{'pre','post'})
    ylabel('Regression coefficient (a.u.)')
    hold on 
    plot(0:5,zeros(6,1),'k')
    plot([ones(48,1)*(2) ones(48,1)*(3)]',(squeeze(beta(:,session,:,2)))','k')
    
end

[h,p, int, stats] = ttest(beta(:,2,1,2),beta(:,2,2,2))
value_pos = ([beta(:,1,1,2),beta(:,1,2,2), beta(:,2,1,2),beta(:,2,2,2)])

%% Plot sorted by value
[~,sorting] = sort(allSubjObject_value);

for session = 1:2
    for p = 1:2
        figure
        set(gcf,'renderer','Painters')
        for s = 1:12
            subplot(3,4,s)
            
            
            % convolve with Gaussian kernel
            scatter(position(:,sorting == s,session,p,1),position(:,sorting == s,session,p,2),'k','filled')
            hold on
            rectangle('Position',[-1,-1,2,2]*radius,'Curvature',[1,1], 'FaceColor','none','EdgeColor','k')
            
            scatter(p_obj(sorting == s,1),p_obj(sorting == s,3),'*y')
            
            axis equal off
            
        end
        set(gcf,'color','w')
    end
end

