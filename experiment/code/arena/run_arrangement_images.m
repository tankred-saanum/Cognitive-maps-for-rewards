%
% original version: 2008
%
% citation: Kriegeskorte N, Mur M (2012) Inverse MDS: inferring dissimilarity structure from multiple item arrangements. Frontiers in psychology, 3.
% clc
% close all hidden;
% 
% options.subjectID = input('Subject ID?      ');
% options.subjectInitials = input('Initials?      ','s');
% 
% addpath('\\mh-fil02.win.ntnu.no\kin\doeller\Mona\ChoiceMaps\jsonlab-1.5')
% % options.root = '\\mh-fil02.win.ntnu.no\kin\doeller\Mona\ChoiceMaps\experiment\';
% options.root = 'C:\Users\mgarvert\ownCloudCBS\Kavli\ChoiceMaps\experiment\';
% 
% data = loadjson(([options.root,'\Data\data_',num2str(options.subjectID),'_2_',options.subjectInitials,'.txt']));
%             
% options.savePath                = [options.root,'datafiles\Subj_',num2str(options.subjectID),'/arena'];
% mkdir (options.savePath)
% 
% addpath ([options.root,'arena/multi-arrangement-2017']);
% 
returnHere = pwd;
% codePath = fullfile(returnHere,'multi-arrangement-2017');
% addpath(genpath(codePath))
%% control variables
options.maxNitemsPerTrial=40; % upper bound on the number of stimuli to be presented in any trial

options.simulationMode=false;
% true: arrangements simulated by MDS for randomly generated high-dimensional ground-truth RDM
%       this enables us to assess whether multiarrangement is likely to
%       outperform pairwise judgments in terms of accuracy in a given
%       scenario (e.g. for 60 items in total, with maximally 40 items per
%       trial)
% false: human subject arranges the items by drag and drop

options.maxSessionLength_min=.3;
options.axisUnits='normalized'; % images resized


%% get subject initials                                         
% options.subjectInitials=inputdlg('Subject initials:');
% options.subjectInitials=options.subjectInitials{1};
options.subjectInitials=data.subject;


%% load object icons

% Load stimuli
nItems= length(data.stimuli); % up to 92 here, or whatever the number of object icons provided
for i = 1:nItems
    theImageLocation{i}= [options.root, sprintf('images/obj%02d.png',data.stimuli(i))];
    [theImage{i},~,transparency{i}] = imread(theImageLocation{i},'BackgroundColor','none');
    options.bgColor = [0.8 0.8 0.8];% [0.7852 0.4414 0.4336; 0.4353 0.6745 0.7843];
end

% create alpha channels for transparent backgrounds
for itemI=1:nItems
    stimuli(itemI).image = theImage{itemI};
    stimuli(itemI).alpha=transparency{itemI};%1-(sqrt(sum((double(theImage{itemI})-255).^2,3))<9);
end

%% prepare output directory
files=dir('similarityJudgementData');
if size(files,1)==0
    % folder 'similarityJudgementData' doesn't exist within current folder: make it
    mkdir('similarityJudgementData');  
end


%% administer session
options.dateAndTime_start=clock;
options.saveEachTrial = 1;

% MULTI-ARRANGEMENT (MA)
%[estimate_dissimMat_ltv_MA,simulationResults_ignore,story_MA]=simJudgmentByMultiArrangement_circArena_liftTheWeakest(stimuli,'Please arrange these objects according to their similarity',options);


options.figname = [options.dataPath,'/Arena_Subj_',num2str(data.subject)];
options.type = 'similarity';
[estimate_dissimMat_ltv_MA_similarity,~,story_MA_similarity,data,arena]=simJudgeByMultiArrangement_circArena_ltw_maxNitems(stimuli,'Bitte ordnen Sie die Monster nach Ähnlichkeit, so dass ähnliche Monster nahe beieinander liegen.',options,data);
copyfile([options.dataPath,'/Arena_Subj_',data.subject,'.fig'],[options.dataPath,'/Arena_Subj_',data.subject,'_similarity.fig'])

figs=openfig([options.dataPath,'/Arena_Subj_',data.subject,'_similarity.fig']);
print(figs,[options.dataPath,'/Arena_Subj_',data.subject,'_similarity.png'],'-dpng')

data.arena_similarity = arena;

options.figname = [options.dataPath,'/Arena_Subj_',num2str(data.subject)];
options.type = 'space';
[estimate_dissimMat_ltv_MA_space,~,story_MA_space,data,arena]=simJudgeByMultiArrangement_circArena_ltw_maxNitems(stimuli,'Bitte ordnen Sie die Monster nach ihrer Position in der Arena.',options,data);
copyfile([options.dataPath,'/Arena_Subj_',data.subject,'.fig'],[options.dataPath,'/Arena_Subj_',data.subject,'_space.fig'])

figs=openfig([options.dataPath,'/Arena_Subj_',data.subject,'_space.fig']);
print(figs,[options.dataPath,'/Arena_Subj_',data.subject,'_space.png'],'-dpng')

data.arena_space = arena;

%% save experimental data from the current subject
save(fullfile(options.dataPath,'arena_workspace'),'*');
cd(returnHere)