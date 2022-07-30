%% Merge two seperate vizard files if Vizard had to be started from scratch
%
% Mona Garvert 1 - 2019
%__________________________________________________________________________
% Copyright (C) 2018 CBI MPS


toSaveAs = 'data_1_1_ih_viz.txt';
toLoad1 = 'data_1_1_ih_viz_1.txt';
toLoad2 = 'data_1_1_ih_viz_2.txt';


%% Update accordingly
data1 = loadjson(['/data/g_gr_doeller-share/Experiments/Mona/ChoiceMaps/experiment/version_2/datafiles/Subj_1/',toLoad1]);
data2 = loadjson(['/data/g_gr_doeller-share/Experiments/Mona/ChoiceMaps/experiment/version_2/datafiles/Subj_1/',toLoad2]);


%% Change this as appropriate
data1.session_1.positionObject.run_6 = data2.session_1.positionObject.run_6
data1.session_1.positionObject.run_7 = data2.session_1.positionObject.run_7
data1.session_1.positionObject.run_8 = data2.session_1.positionObject.run_8
data1.session_1.positionObject.run_9 = data2.session_1.positionObject.run_9
data1.session_1.positionObject.run_10 = data2.session_1.positionObject.run_10

data1.session_1.freeExplore.run_6 = data2.session_1.freeExplore.run_6
data1.session_1.freeExplore.run_7 = data2.session_1.freeExplore.run_7
data1.session_1.freeExplore.run_8 = data2.session_1.freeExplore.run_8
data1.session_1.freeExplore.run_9 = data2.session_1.freeExplore.run_9
data1.session_1.freeExplore.run_10 = data2.session_1.freeExplore.run_10

data = data1;

%% Convert to json and save as text file
S = jsonencode(data);

fid = fopen(['/data/g_gr_doeller-share/Experiments/Mona/ChoiceMaps/experiment/version_2/datafiles/Subj_1/',toSaveAs], 'w');
if fid == -1, error('Cannot create txt file'); end
fwrite(fid, S, 'char');
fclose(fid);

%% Doublecheck whether it worked
inspect_data = loadjson(['/data/g_gr_doeller-share/Experiments/Mona/ChoiceMaps/experiment/version_2/datafiles/Subj_1/',toSaveAs]);
