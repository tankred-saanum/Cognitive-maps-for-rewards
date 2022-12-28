%% Script used for analysing behavioural data of the ChoiceMap taslk


clear all
close all

% Define baseline directory
bdir = '//data/p_02071/choice-maps/Cognitive-maps-for-rewards/';

% Add dependencies
addpath(genpath([bdir,'/analysis/helper_scripts']))


% Analyse population data
savedir = [bdir,'/behavior/datafiles/merged_data'];

% plot map structure
startSubj = 101;
load([savedir,'/subj_',num2str(startSubj),'/data_',num2str(startSubj),'.mat']);
inference_objects = data.mat{3}.data.options.inference_objects;

removeID = [21,36,37,38]; % individuals to remove due to technical problems during scanning

table = [];

load([bdir,'/figures/data/population_data.mat']);
stimulus_value = [all_data.real_value(1:12); all_data.real_value(13:24)];


subjList = 101:152;
subjList([21,36,37,38]) = [];

for c = 1:length(subjList)
    subj = subjList(c);

    disp(['Subject ', num2str(subj)])
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);

    p_obj = data.mat{2}.data.objPositions; % object location
    distance = squareform(pdist(p_obj));    % distance between pairs of object

    for session = 1:2
        trial = [];
        cr = [];
        RT = [];
        probe = [];
        option1 = [];
        option2 = [];
        map = [];
        dtrial = [];
        vtrial = [];
        side = [];
        valuediff1 = [];
        valuediff2 = [];
        distdiff1 = [];
        distdiff2 = [];
        distoptions = [];

        % values
        probevalue   = [];
        optionvalue1  = [];
        optionvalue2  = [];


        for run = 1:3

            od = data.mat{session+1}.data.scan{run}.RSA.objDiff;
            choice_too_fast = od.RT < 0.3;
            od.choicetrial(choice_too_fast) = 0;    % sometimes the choices were made automatically by the button press. ignore these.

            % create csv with all the relevant data
            trial   = [trial find(od.choicetrial==1)];
            cr      = [cr; od.cr(od.choicetrial==1)];
            RT      = [RT; od.RT(od.choicetrial==1)];
            probe   = [probe; od.stimForChoice(od.choicetrial==1)];
            option1 = [option1; od.choiceOptions(od.choicetrial==1,1)];
            option2 = [option2; od.choiceOptions(od.choicetrial==1,2)];
            map     = [map od.map(od.choicetrial==1)];
            dtrial  = [dtrial strcmp(od.whichchoice(od.choicetrial==1),'d')];
            vtrial  = [vtrial strcmp(od.whichchoice(od.choicetrial==1),'v')];
            side    = [side od.choice(od.choicetrial==1)];

            %od.choice == 1: left key
            %od.choice == 2: right key;

            for i = 1:length(probe)
                % distances
                distdiff1(length(distdiff1)+1) = distance(probe(i),option1(i));
                distdiff2(length(distdiff2)+1) = distance(probe(i),option2(i));
                distoptions(length(distoptions)+1) = distance(option1(i),option2(i));

                % values
                probevalue(length(probevalue)+1)   = stimulus_value(map(i),probe(i));
                optionvalue1(length(optionvalue1)+1)    = stimulus_value(map(i),option1(i));
                optionvalue2(length(optionvalue2)+1)    = stimulus_value(map(i),option2(i));
            end
            valuediff1 = optionvalue1-probevalue;
            valuediff2 = optionvalue2-probevalue;
            valuediffoptions = optionvalue1-optionvalue2;



        end



        % regression for distance trials
        ix = dtrial==1;
        dm = zscore([ones(sum(ix),1)  distdiff1(ix)' distdiff2(ix)' abs(valuediff1(ix))' abs(valuediff2(ix))' optionvalue1(ix)' optionvalue2(ix)']);

        cr_ols(1,c,session, :) = ols(cr(ix), dm, eye(size(dm,2)));
        RT_ols(1,c,session, :) = ols(RT(ix), dm, eye(size(dm,2)));
        %             side_ols(1,c,session, :) = ols(side(ix)', dm, eye(size(dm,2)));
        [B(1,c,session, :),dev,stats] = mnrfit(dm(:,2:end),(side(ix))');

        % regression for value trials
        if session == 2
            ix = vtrial==1;
            dm = [ones(sum(ix),1) zscore([ distdiff1(ix)' distdiff2(ix)' abs(valuediff1(ix))' abs(valuediff2(ix))' optionvalue1(ix)' optionvalue2(ix)'])];

            cr_ols(2,c,session, :) = ols(cr(ix), dm, eye(size(dm,2)));
            RT_ols(2,c,session, :) = ols(RT(ix), dm, eye(size(dm,2)));
            %                 side_ols(2,c,session, :) = ols(side(ix)', dm, eye(size(dm,2)));
            %
            [B(2,c,session, :),dev,stats] = mnrfit(dm(:,2:end),(side(ix))');

        end
    end
end


%%
side_ols=B;
plotix = [1:6 8:13]*2-1;
figure('Renderer', 'painters', 'Position', [10 10 1200 400])
for session = 1:2
    hold on

    % Diffculty
    cr_plot = squeeze(nanmean(cr_ols(:,:,session,2:end),2))'; cr_std = squeeze(nanstd(cr_ols(:,:,session,2:end),0,2))'/sqrt(48);
    RT_plot = squeeze(nanmean(RT_ols(:,:,session,2:end),2))'; RT_std = squeeze(nanstd(RT_ols(:,:,session,2:end),0,2))'/sqrt(48);
    side_plot = squeeze(nanmean(side_ols(:,:,session,2:end),2))'; side_std = squeeze(nanstd(side_ols(:,:,session,2:end),0,2))'/sqrt(48);
    
    % %
    individual_data = [squeeze(side_ols(1,:,session,2:end)) squeeze(side_ols(2,:,session,2:end))];
    subplot(1,2,session)
    [h,p,x,stats] = ttest([squeeze(side_ols(1,:,session,2:end)) squeeze(side_ols(2,:,session,2:end))]);
    boxplot([squeeze(side_ols(1,:,session,2:end)) squeeze(side_ols(2,:,session,2:end))],'positions',plotix,'colorgroup',h==1, 'colors',[0.5 0.5 0.5;0.8 0 0]);
    prepImg
    ylabel('Parameter estimate')

    hold on
    for i = 1:12
        scatter(repmat(plotix(i)+0.5,1,48),(individual_data(:,i)),'k')
    end


end

