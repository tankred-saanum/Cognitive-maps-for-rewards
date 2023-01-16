clear all
close all

% Define baseline directory
bdir = '/data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

% Add dependencies
addpath(genpath([bdir,'/figures/helper_scripts']))


% Analyse population data
savedir = [bdir,'/behavior/datafiles/merged_data'];

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning


allvif = nan(52,2,123);

for subj = 101:152
    for session = 2:3
        try
        load ([bdir,'/figures/VIF/781_',num2str(subj),'_',num2str(session),'_vif.mat'])
        

        allvif(subj-100,session-1,:) = vif.allvifs;
        catch
            allvif(subj-100,session-1,:) = nan;
        
        end
        
    end
end

for i = 1:123
    ix(i) = contains(vif.name{i},'experienced_distance');
end
vifS = squeeze(nanmean(allvif(:,:,ix==1),3));
vifS(removeID,:,:) = [];

index = 101:152; index(removeID) = [];

for s = 1:2
    exclude{s} = find(vifS(:,s) > 4);
end

excludeID = index(exclude{2});

%%
figure('Renderer', 'painters', 'Position', [10 10 900 400])
subplot(1,3,1)
bar(sort(vifS(:,2,:))')
prepImg
xlabel('Participant')
ylabel('VIF')
yline(5,'-','Threshold');
xlim([0 49])

%% Load imaging data
% Define baseline directory

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
    pe_spatial(counter) = load(fullfile(bdir,'figures','masks',roi,['session_',session],'03_allDistance_s8',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'03_allDistance_s8')));
    pe_temporal(counter) = load(fullfile(bdir,'figures','masks',roi,['session_',session],'04_SRDistance_s8',sprintf('%d_%s_%s_%s.txt',subj,session,roi,'04_SRDistance_s8')));
end

%%

subplot(1,3,2)
scatter(vifS(:,2,:),pe_spatial,'filled'), lsline
[r,p,RL,RU] = corrcoef(vifS(:,2,:),pe_spatial');
xlabel('VIF')
ylabel('Spatial fMRI effect')
title(sprintf('r = %.2f, p = %.2f',r,p));
prepImg


subplot(1,3,3); 
scatter(vifS(:,2,:),pe_temporal,'filled'), lsline
[r,p,RL,RU] = corrcoef(vifS(:,2,:),pe_temporal')
xlabel('VIF')
ylabel('Temporal fMRI effect')
title(sprintf('r = %.2f, p = %.2f',r,p));
prepImg



