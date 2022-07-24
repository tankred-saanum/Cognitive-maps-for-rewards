%% Choice task
% pick each graph location exactly once for a choice trial (either on map 1
% or on map 2)

disp('Generating stimulus sequence... ')

% make sure that across the three blocks each object is probed twice, once
% for distance, once for value. This means 24*2/3 = 16 choices per block
all_choices = randperm(options.nodes*4);
all_choices_stim = mod(all_choices,12);
    
while sum(all_choices(1:16)<=options.nodes*2)~=8 || sum(all_choices(17:32)<=options.nodes*2)~=8 || sum(all_choices(33:48)<=options.nodes*2)~=8 ... % Equal number of distance and value trials
        || any (histcounts(all_choices_stim(1:16),0:12) > 2) || any (histcounts(all_choices_stim(17:32),0:12) > 2) || any (histcounts(all_choices_stim(33:48),0:12) > 2)... % make sure an object is not presented more than twice in a block
    all_choices = randperm(options.nodes*4);
    all_choices_stim = mod(all_choices,12); 
end

      


start_obj   = [1 17 33];
end_obj     = [16 32 48];

if data.session == 3
    choicetype      = all_choices<=options.nodes*2;  % 0: dist, 1: value
else
    choicetype      = zeros(1,length(all_choices)); 
end
choiceobject    = mod(all_choices,24); choiceobject(choiceobject == 0) = 24; % <=12: context 1, > 12: context 2

totalMax = zeros(options.nodes*2);
for run = 1:options.scanblocks
    disp(['Block ',num2str(run)])
    counter = 1;
    
    data.scan{run}.RSA.objDiff.choicetrial          = zeros(1,options.scanblocklength);
    data.scan{run}.RSA.objDiff.stimForChoice        = zeros(options.scanblocklength,1);
    data.scan{run}.RSA.objDiff.choiceOptions        = zeros(options.scanblocklength,2);
    data.scan{run}.RSA.objDiff.('v'){1}             = zeros(options.scanblocklength,3);
    data.scan{run}.RSA.objDiff.('v'){2}             = zeros(options.scanblocklength,3);    
    data.scan{run}.RSA.objDiff.('d')                = zeros(options.scanblocklength,3);
    data.scan{run}.RSA.objDiff.map                  = (data.settings.seq{run}(1,:)>options.nodes)+1;    
    data.scan{run}.RSA.objDiff.whichchoice          = strings(1,length(sq.seq{run}));
    
            
    for obj = start_obj(run):end_obj(run)
        ix = find(data.settings.seq{run}(1,:) == choiceobject(obj));
        random_pick = ix(randi(length(ix)));
        data.scan{run}.RSA.objDiff.choicetrial(random_pick) = 1;       % Trials where participants have to make a decision
        if choicetype(obj) == 0
            data.scan{run}.RSA.objDiff.whichchoice(random_pick) = 'd';
        else
            data.scan{run}.RSA.objDiff.whichchoice(random_pick) = 'v';
        end
    end
    
    % choose orthogonal choices on choice trials
    % Pick two random objects
    % Needs to be replaced such that distances are orthogonalised!

    choicetrials    = find(~strcmp( data.scan{run}.RSA.objDiff.whichchoice,''));
    mapping         = double(data.settings.seq{run}(1,choicetrials)>options.nodes)+1;
    
    keepGoing = 1;
    while keepGoing   
        if counter == 1
            %        Pick a set of stimuli such that each position is used twice
            choiceOptions         = [[1:options.nodes 1:options.nodes];[1:options.nodes 1:options.nodes]]';
            choiceOptions(:,1)    = choiceOptions(randperm(length(choiceOptions)),1);
            choiceOptions(:,2)    = choiceOptions(randperm(length(choiceOptions)),2);

            choiceOptions = choiceOptions(1:16,:);
            
            %         Pick chosen stimulus
            stimForChoice       = data.settings.seq{run}(1,choicetrials)';               
            choiceOptions_old   = choiceOptions;
        else
            choiceOptions = choiceOptions_old;
            flip = randi(length(choiceOptions),2);
            choiceOptions(flip(:,1),1) = choiceOptions(flip(end:-1:1,1),1);
            choiceOptions(flip(:,2),2) = choiceOptions(flip(end:-1:1,2),2);
            choiceOptions = choiceOptions(1:16,:);
            
        end
        stimForChoice(stimForChoice > options.nodes) = stimForChoice(stimForChoice > options.nodes) - options.nodes;
            
        
        for i = 1:length(stimForChoice)
            spatialDist(i,1) = data.settings.spatial_dist(stimForChoice(i),choiceOptions(i,1));
            spatialDist(i,2) = data.settings.spatial_dist(stimForChoice(i),choiceOptions(i,2));
            spatialDist(i,3) = data.settings.spatial_dist(choiceOptions(i,1),choiceOptions(i,2));

            valueDist{1}(i,1) = data.settings.value_dist{1}(stimForChoice(i),choiceOptions(i,1));
            valueDist{1}(i,2) = data.settings.value_dist{1}(stimForChoice(i),choiceOptions(i,2));
            valueDist{1}(i,3) = data.settings.value_dist{1}(choiceOptions(i,1),choiceOptions(i,2));

            valueDist{2}(i,1) = data.settings.value_dist{2}(stimForChoice(i),choiceOptions(i,1));
            valueDist{2}(i,2) = data.settings.value_dist{2}(stimForChoice(i),choiceOptions(i,2));
            valueDist{2}(i,3) = data.settings.value_dist{2}(choiceOptions(i,1),choiceOptions(i,2));
        end

        relValDist(mapping==1,:) = valueDist{1}(mapping==1,:); 
        relValDist(mapping==2,:) = valueDist{2}(mapping==2,:);
        irrelValDist(mapping==1,:) = valueDist{2}(mapping==1,:); 
        irrelValDist(mapping==2,:) = valueDist{1}(mapping==2,:);
        
        cr = corr([spatialDist relValDist irrelValDist]);
        if counter == 1
            cr_old = cr;
        end

        if (~any((choiceOptions(:,1) - choiceOptions(:,2)) == 0) ...
                && ~any(stimForChoice == choiceOptions(:,1)) ...
                && ~any(stimForChoice == choiceOptions(:,2))) ...
                && ~any(abs(cr(eye(length(cr))==0))>0.25)
            % if the new one is better than the old one and all conditions
            % are fulfilled stop the loop
            keepGoing = 0;
            
        elseif ~(~any((choiceOptions(:,1) - choiceOptions(:,2)) == 0) ...
                && ~any(stimForChoice == choiceOptions(:,1)) ...
                && ~any(stimForChoice == choiceOptions(:,2)))
            
            choiceOptions         = [[1:options.nodes 1:options.nodes];[1:options.nodes 1:options.nodes]]';
            choiceOptions(:,1)    = choiceOptions(randperm(length(choiceOptions)),1);
            choiceOptions(:,2)    = choiceOptions(randperm(length(choiceOptions)),2);
    
        elseif   max (abs(cr(eye(length(cr))==0))) > max(abs(cr_old(eye(length(cr_old))==0)))
            
            % if the old one is better than the new one then try again
            keepGoing = 1;
            counter = counter+1;
        else            
            
            % if the new one is better than the old one then update, but
            % only if none of the other conditions is violated
            keepGoing = 1;
            counter = counter+1;
            
            if (~any((choiceOptions(:,1) - choiceOptions(:,2)) == 0) ...
                && ~any(stimForChoice == choiceOptions(:,1)) ...
                && ~any(stimForChoice == choiceOptions(:,2)))
            
%                 disp([max(abs(cr(eye(length(cr))==0))) max(abs(cr_old(eye(length(cr_old))==0)))])
                choiceOptions_old = choiceOptions;
                cr_old = cr;    
                
            end
        end
    end
%    disp([max(abs(cr(eye(length(cr))==0))) max(abs(cr_old(eye(length(cr_old))==0)))])
                
    data.scan{run}.RSA.objDiff.stimForChoice(choicetrials)      = stimForChoice;
    data.scan{run}.RSA.objDiff.choiceOptions(choicetrials,:)    = choiceOptions;
    
    data.scan{run}.RSA.objDiff.('v'){1}(choicetrials,:)         = valueDist{1};
    data.scan{run}.RSA.objDiff.('v'){2}(choicetrials,:)         = valueDist{2};
    data.scan{run}.RSA.objDiff.('d')(choicetrials,:)            = spatialDist;
    
    data.scan{run}.RSA.objDiff.relValueDist(choicetrials,:)     = relValDist;    
    data.scan{run}.RSA.objDiff.irrelValueDist(choicetrials,:)   = irrelValDist;    
    data.scan{run}.RSA.objDiff.jitter(choicetrials)             = random(options.ITI,length(choicetrials),1);  % mean: 2.2
end


