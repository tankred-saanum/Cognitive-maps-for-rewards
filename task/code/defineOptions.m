%  defineOptions is a nullary function which initialises a struct
%  containing the preferences and details for the experiment.
%  A new data structure will be created for each subject.
%
%  Mona Garvert 10 - 2018
%__________________________________________________________________________
% Copyright (C) 2018 CBI MPS

% Initialise seed
s = RandStream.create('mt19937ar','Seed',sum(100*clock));
RandStream.setGlobalStream(s);

%% Project details
% =========================================================================
% Change these if necessary
options.test                = false;
options.version             = 'scan'; % Experimental version
options.scan                = 1;        % Scan or not? 
options.cbs                 = true;

% Leave these untouched
options.root                = ['C:/Users/user/Documents/Experiments/Mona/ChoiceMaps/version_',num2str(options.version),'/'];
options.imagePath           = [options.root 'images/'];
options.nodes               = 12;       % Number of nodes on the graph
options.scanblocks          = 3;
options.scanblocklength     = 144;      % Number of trials in each block
options.choiceblocklength   = 100;
options.inference_objects   = [5 11; 12 6];

addpath(genpath(options.root))

%%
if options.test
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    disp('Running in test mode!! Change options.test for real experiment!')
    options.screensize          = 1;    % 0 for small window, 1 for full screen
    PsychDebugWindowConfiguration
else
    options.screensize          = 1;    % 0 for small window, 1 for full screen
end

subjNo          = input('Participant ID?    ');
initials        = input('Participant initials?     ','s');
session         = input('Experimental session?     ');
scan_session    = input('Scan session?     ');

options.scanblocks              = 3;    % RSA task
if scan_session == 1                    % Choice task only before the first scan session
    options.choiceblocks        = 1;
else
    options.choiceblocks        = 0;
end

% Load existing data file
options.dataPath        = [options.root 'datafiles/Subj_' num2str(subjNo), '/session_',num2str(session)];       % Save data under current day
mkdir(options.dataPath);

% data            = loadjson([options.dataPath,'/data_',num2str(subjNo),'_',num2str(data.session-1),'_',initials,'_viz.txt']);
data            = loadjson([options.dataPath,'/data_',num2str(subjNo),'_',num2str(session),'_pre_',initials,'_viz.txt']);
data.day        = input('Day since first day of training?    ');

col = ([102,146,166;...
    169,111,111]-50)/255;

if mod(subjNo,2) == 0
    data.context = col;
    data.contextLabel = {'b','r'};
else
    data.context = col(2:-1:1,:);
    data.contextLabel = {'r','b'};
end


%% Scan details / port input
% =====================================================================================================
data.scanTech.buttonPort     = 'COM1';
data.scanTech.triggerPort    = 'COM7';
data.scanTech.dummy          = 5;        % no. of dummy vols

data.options = options;


%%
% Generate jitter
options.ITI          = makedist('Exponential');
options.ITI_RSA      = makedist('Exponential');
if options.scan || options.cbs
    options.ITI.mu       = 3;
    options.ITI          = truncate(options.ITI,2,5);
else
    options.ITI.mu       = 1.2;
    options.ITI          = truncate(options.ITI,1,1.5);
end

% load([options.root, '/expt_specs/seq_',num2str(subj),'.mat']);
data.choice.jitter      = random(options.ITI,options.choiceblocklength,1);  % mean: 2.2
data.choice.fb_jitter   = random(options.ITI,options.choiceblocklength,1);  % mean: 2.2
happy          = (sum(data.choice.jitter ) + sum(data.choice.fb_jitter ));

while happy > (options.ITI.mu  *2*options.choiceblocklength) * 1.1
    data.choice.jitter        = random(options.ITI,options.choiceblocklength,1);  % mean: 2.2
    data.choice.fb_jitter     = random(options.ITI,options.choiceblocklength,1);  % mean: 2.2
    happy = (sum(data.choice.jitter ) + sum(data.choice.fb_jitter ));
end

restart          = input('Neustart des Objekt-Teils nach Unterbrechung? (j/n)    ','s');
if strcmp(restart,'j')
    startblock          = input('Start mit welchem Block?    ');    
    saveSes = 1;
    fname = ['/data_',data.subject,'_',num2str(session),'_',initials,'.mat'];
    load([options.dataPath,fname]);
    
    fname = ['/data_',data.subject,'_',num2str(session),'_',initials,'_sess',num2str(saveSes),'.mat'];
    while exist([options.dataPath,fname],'file')
        load([options.dataPath,fname]);
        saveSes = saveSes + 1;
        fname = ['/data_',data.subject,'_',num2str(session),'_',initials,'_sess',num2str(saveSes),'.mat'];
    end
else
    startblock          = 1;
    fname = ['/data_',data.subject,'_',num2str(session),'_',initials,'.mat'];
    if ~options.test && exist([options.dataPath,fname],'file')
        error('File for this subject exists already! aborting...')
    end
end

%%   Update this!
load([options.root, '/expt_specs/seq_',data.subject,'_sess',num2str(session-1),'.mat']);
data.settings           = sq;
data.session            = session;

% Set up choice trials during RSA task
setRSATask

for bl = startblock:3
    data.scan{bl}.RSA.jitter    = random(options.ITI,options.scanblocklength,1);  % mean: 2.2
end
