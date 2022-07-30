%% Supplementary Figures 1
clear
close all

bdir = '/data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

addpath (genpath([bdir,'/figures/helper_scripts']))
savedir = [bdir,'behavior/datafiles/merged_data'];

subjList = 101:152;
subjList([21,36,37,38]) = [];

figure
set(gcf,'renderer','Painters')


hexcol = 'BA9CD1';
cm = [186 156 209]/255;

tiledlayout(8,6, 'Padding', 'none', 'TileSpacing', 'compact'); 
for c=1:length(subjList)
    subj = subjList(c);
    nexttile    
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
    load([savedir,'/subj_',num2str(subj),'/sub-',num2str(subj),'_exploration_data.mat'])
    plot(explorationPath(:,1),explorationPath(:,2),'k');
    hold on
    rectangle('Position',[-1,-1,2,2]*15,'Curvature',[1,1], 'FaceColor','none','EdgeColor','k')
    
    % plot object locations
    p_obj = data.mat{2}.data.objPositions*15;
    scatter(p_obj(:,1),p_obj(:,3),50,cm(end,:),'filled')
    xlim([-15 15])
    ylim([-15 15])
    axis equal off
    prepImg
    
    pathlength(c) = length(explorationPath);

end
