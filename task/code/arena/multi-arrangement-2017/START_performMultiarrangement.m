% START_performMultiarrangement
%
% original version: 2008
% this version: April 2015 (control maximum number of stimuli per trial using options.maxNitemsPerTrial)
%
% citation: Kriegeskorte N, Mur M (2012) Inverse MDS: inferring dissimilarity structure from multiple item arrangements. Frontiers in psychology, 3.

clear;clc
close all hidden;


%% control variables
nItems=12; % up to 92 here, or whatever the number of object icons provided
options.maxNitemsPerTrial=80; % upper bound on the number of stimuli to be presented in any trial

options.simulationMode=false;
% true: arrangements simulated by MDS for randomly generated high-dimensional ground-truth RDM
%       this enables us to assess whether multiarrangement is likely to
%       outperform pairwise judgments in terms of accuracy in a given
%       scenario (e.g. for 60 items in total, with maximally 40 items per
%       trial)
% false: human subject arranges the items by drag and drop

options.maxSessionLength_min=60;
options.sessionI=1;  
options.axisUnits='normalized'; % images resized


%% get subject initials                                         
% options.subjectInitials=inputdlg('Subject initials:');
% options.subjectInitials=options.subjectInitials{1};
options.subjectInitials='blah';


%% load object icons
load('stimuli.mat');
% load('stimuli_92objs.mat');
% stimuli=stimuli_92objs(1:nItems);
stimuli=stimuli(1:nItems);
% create alpha channels for transparent backgrounds
for itemI=1:nItems
    stimuli(itemI).alpha=1-(sqrt(sum((double(stimuli(itemI).image)-128).^2,3))<9);
end


%% prepare output directory
files=dir('similarityJudgementData');
if size(files,1)==0
    % folder 'similarityJudgementData' doesn't exist within current folder: make it
    mkdir('similarityJudgementData');  
end


%% administer session
options.dateAndTime_start=clock;

% MULTI-ARRANGEMENT (MA)
%[estimate_dissimMat_ltv_MA,simulationResults_ignore,story_MA]=simJudgmentByMultiArrangement_circArena_liftTheWeakest(stimuli,'Please arrange these objects according to their similarity',options);
[estimate_dissimMat_ltv_MA,simulationResults_ignore,story_MA]=simJudgeByMultiArrangement_circArena_ltw_maxNitems(stimuli,'Please arrange these objects according to their similarity',options);


%% save experimental data from the current subject
save(['similarityJudgementData\',options.subjectInitials,'_session',num2str(options.sessionI),'_',dateAndTimeStringNoSec_nk,'_workspace']);


%% display representational dissimilarity matrix (RDM)
showRDMs(estimate_dissimMat_ltv_MA);
addHeadingAndPrint('multiple-trial RDM','similarityJudgementData\figures');


%% plot stimuli in multidimensional-scaling (MDS) arrangement
criterion='metricstress';
[pats_mds_2D,stress,disparities]=mdscale(estimate_dissimMat_ltv_MA,2,'criterion',criterion);

pageFigure(400); subplot(2,1,1); 
drawImageArrangement(stimuli,pats_mds_2D,1,stimuli(1).image(1,1,:));
title({'\fontsize{14}stimulus images in MDS arrangement\fontsize{11}',[' (',criterion,')']});
shepardPlot(estimate_dissimMat_ltv_MA,disparities,pdist(pats_mds_2D),[400 2 1 2],['\fontsize{14}shepard plot\fontsize{11}',' (',criterion,')']);
addHeadingAndPrint('multiple-trial MDS plot','similarityJudgementData\figures');



%% display message DONE
h=msgbox('You are done :)');

