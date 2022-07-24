%% Many maps experiment
% 
% saved as data_#.mat in ../datafiles
%
%__________________________________________________________________________
% Copyright (C) 2018 Mona M Garvert, MPI CBS Leipzig

% Clear the workspace and the screen
close all;

%% Start cogent and setup experiment
% =========================================================================


%% Project details
% =========================================================================
% Change these if necessary
options.test                = false;
options.version             = 'scan'; % Experimental version
options.scan                = 0;        % Scan or not? 
options.cbs                 = false;

% Leave these untouched
options.root                = ['/data/g_gr_doeller-share/Experiments/Mona/ChoiceMaps/experiment/version_',num2str(options.version),'/'];
options.imagePath           = [options.root 'images/'];

addpath(genpath(options.root))
% addpath(genpath([options.root,'/jsonlab-1.5']))

subjNo          = input('Participant ID?    ');
initials        = input('Participant initials?     ','s');
session         = input('Experimental session?     ');
scan_session    = input('Scan session?     ');

% Load existing data file
options.dataPath        = [options.root 'datafiles/Subj_' num2str(subjNo), '/session_',num2str(session)];       % Save data under current day
mkdir(options.dataPath);

data            = loadjson([options.dataPath,'/data_',num2str(subjNo),'_',num2str(session),'_pre_',initials,'_viz.txt']);
data.day        = input('Day since first day of training?    ');
data.session    = session;
data.initials   = initials;

% if ~options.test && exist([options.dataPath,'/data_',num2str(subjNo),'_',num2str(session),'_',initials,'.mat'],'file')
%     error('File for this subject exists already! aborting...')
% else mkdir (options.dataPath)    
% end 

col = (([152,196,216;219,161,161])-50)/255;

if mod(subjNo,2) == 0
    data.context = col;
    data.contextLabel = {'b','r'};
else
    data.context = col(2:-1:1,:);
    data.contextLabel = {'r','b'};
end

if options.test
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    disp('Running in test mode!! Change options.test for real experiment!')
    options.screensize          = 1;    % 0 for small window, 1 for full screen
%     PsychDebugWindowConfiguration
else
    options.screensize          = 1;    % 0 for small window, 1 for full screen
end

setup

%% Likert task
rectColor = [0.50 0.5 0.5];
% Screen('FillRect', window, rectColor, centeredRect);

% likert_task

%% Likert task
value_rating_task

%% Stop Psychtoolbox
sca;

%     Arena task
run_arrangement_images
save ([options.dataPath,'/data_',data.subject,'_',num2str(session),'_',initials,'_postscan.mat'],'data');

