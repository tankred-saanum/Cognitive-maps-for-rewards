% Convert Matlab data structres into tables that can be used for example in
% python

savedir = ['/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data'];
startSubj=101;
maxSubj     =152;
nSubj   = 52;

%% 1. Object location and value distribution. The same for all participants
% load data
load([savedir,'/subj_',num2str(startSubj),'/data_',num2str(startSubj),'.mat']);

% extract relevant information
p_obj = data.mat{2}.data.objPositions;
x_obj = p_obj(:,1); y_obj = p_obj(:,3);
values = data.mat{3}.data.settings.value';
values_map1 = values(:,1); values_map2 = values(:,2);

inference_objects = data.mat{3}.data.options.inference_objects;
io_map1 = zeros(12,1); io_map1(inference_objects(1,:)) = 1;
io_map2 = zeros(12,1); io_map2(inference_objects(2,:)) = 1;

% save
t = table(x_obj,y_obj,values_map1,values_map2,io_map1,io_map2);
writetable(t,'/data/p_02071/choice-maps/paper/dataAsCSV/obj_pos_value.csv')

%% Choice behaviour

chosenSideMap = [];
for subj = startSubj:maxSubj
    try
        if exist([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat'],'file')
            load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj)]);
            
            choice = data.mat{3}.data.choice;
            choice.decision = (choice.decision-1.5)*2;
            
            % Chosen side as a function of value differences
            a=sortrows([choice.values(1,:)'-choice.values(2,:)' choice.decision']);
            
            edges = [-100:20:100];
            for i = 1:length(edges)-1
                chosenSide(i,subj-100) = nanmean(a(a(:,1)>edges(i) & a(:,1)<=edges(i+1),2));
                for map = 1:2
%                     chosenSideMap(map,i,subj-100) = nanmean(a(a(:,1)>edges(i) & a(:,1)<=edges(i+1) & choice.map'==map,2));
                    chosenSideMap = [chosenSideMap; [subj map edges(i)+10 nanmean(a(a(:,1)>edges(i) & a(:,1)<=edges(i+1) & choice.map'==map,2))]];
                end
            end
         end
    catch
    end
end
subj    = chosenSideMap(:,1);
map     = chosenSideMap(:,2);
edges   = chosenSideMap(:,3);
value   = chosenSideMap(:,4);

% save
t = table(subj,map,edges,value);
writetable(t,'/data/p_02071/choice-maps/paper/dataAsCSV/choice_beh.csv')


%% Likability rating
%%
clear likert_location stimulus
likert_location     = nan(nSubj,12);
value_rating        = nan(nSubj,24);
location_value      = nan(nSubj,3);

for subj = startSubj:maxSubj
    if exist([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat'],'file')
        try
            load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj)]);
            
            if ~isempty(data.likert)
                % Subject three performed the task by entering ratings in a
                % text brox rather than using a slider
                for trial = 0:11
                    likert_location(subj-100,eval(['data.likert.trial_',num2str(trial),'.stim'])) = eval(['data.likert.trial_',num2str(trial),'.position(:,end)']);
                end
                
                for trial = 0:23
                    value_rating(subj-100,eval(['data.value_rating.trial_',num2str(trial),'.stim'])) = 100*eval(['data.value_rating.trial_',num2str(trial),'.position(:,end)']);
                end
                
                likert_location(subj-100,:) = (likert_location(subj-100,:));
                        
                
                dm = [ones(12,1) value_rating(subj-100,1:12)' + value_rating(subj-100,13:24)'];
                regress_location_value(subj-100,:) = regress(likert_location(subj-100,:)',dm);
                
            end
            catch
                    regress_location_value(subj-100,:) = nan;
        end
        end
                    
end
        
% Create  table
subj = (101:152)';
Tsubj = array2table(subj);
Tlik = array2table(likert_location);
Tregr = array2table(regress_location_value);
Tvalue_rating = array2table(value_rating);

inference_rating = [nanmean(value_rating(:,inference_objects(:,1)),2) nanmean(value_rating(:,inference_objects(:,2)),2)];
Tinference_rating = array2table(inference_rating);
t = [Tsubj Tlik Tregr Tvalue_rating, Tinference_rating];
writetable(t,'/data/p_02071/choice-maps/paper/dataAsCSV/likert.csv')

%% 6. arena
r_arena_true = all_data.r_arena_true';
Tr_arena_true    = array2table(r_arena_true);
r_arena_true_noOutliers = r_arena_true;
r_arena_true_noOutliers(r_arena_true_noOutliers<0.5) = nan;
Tr_arena_true_noOutliers = array2table(r_arena_true_noOutliers);
t = [Tsubj Tr_arena_true, Tinference_rating, Tr_arena_true_noOutliers];
writetable(t,'/data/p_02071/choice-maps/paper/dataAsCSV/arena.csv')

