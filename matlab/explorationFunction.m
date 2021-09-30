savedir = 'zipmapdata/';
subStart = 101;
subStop = 152;

exclude = [136, 137, 138, 121];
% 131, 132, 133, 134, 135, -- 137, 138 (both excluded) -- , 140, 141



for subj = subStart:subStop
    if ~ ismember(subj, exclude)
        exploration(subj, savedir)
    end
end


function exploration(subj, savedir)
%for subj = subStart:subStop
    load([savedir,'sub-',num2str(subj),'/data_',num2str(subj)]);
    done ='';

    % distance to object
    limit = 3;

    % object positions. 15 is the radius
    p_obj = data.mat{2}.data.objPositions*15;
    expl = [];
    session=1;
    run = 1;
    alldistanceRun = [];
    explallRun = [];
    
    while isfield(eval(['data.viz.session_', num2str(session),'.pre.freeExplore']),['run_',num2str(run)]) && ~isempty(eval(['data.viz.session_', num2str(session),'.pre.freeExplore.run_',num2str(run)])) % if not empty
        explRun{run} = [];

        % d.position contains the position of the agent
        % at each point in time. 
        d = eval(['data.viz.session_', num2str(session),'.pre.freeExplore.run_',num2str(run)]);

        % do this separately for each run
        explRun{run} = d.position(:,[1,3]);%/(data.viz.radius*2);
        run = run+1;
    end

    for run = 1:numel(explRun)  % go through all exploration runs and compute distance to each object at each timepoint
        for obj = 1:12
            alldistanceRun{run}(:,obj) = pdist2(explRun{run}(:,[1 2]),p_obj(obj,[1 3]));
        end
    end

    % Do this separately for each run
    for run = 1:numel(explRun)  % go through all exploration runs

        explallRun{run} = [];
        explallRun{run}(:,1:2)=  explRun{run}(:,1:2);
        for i = 1:size(explallRun{run},1)

            % if distance is smaller than a given limit, then find the object
            % that is the closest to the current position
            if min(alldistanceRun{run}(i,:))<limit
                explallRun{run}(i,3) = find(alldistanceRun{run}(i,:) == min(alldistanceRun{run}(i,:)));
            else explallRun{run}(i,3) = 0;
            end



        end

    end
    folderName = ['sub_expl_', num2str(subj)];
    
    mkdir (['sub_expl_', num2str(subj)])
    
    %display(run);
    %display(subj);
    for fileNum = 1:run
       fileName = [folderName, '/expl_', num2str(fileNum), '.csv'];
       csvwrite(fileName,explallRun{fileNum})
       %writematrix(explallRun{fileNum}, fileName);
    end
end
