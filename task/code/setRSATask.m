%% Choice task
% pick each graph location exactly once for a choice trial (either on map 1
% or on map 2)

disp('Generating stimulus sequence... ')

totalMax = zeros(options.nodes*2);
for run = startblock:options.scanblocks
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
    
            
    for obj = 1:options.nodes*2
        ix = find(data.settings.seq{run}(1,:) == obj);
        data.scan{run}.RSA.objDiff.choicetrial(ix(randi(length(ix)))) = 1;       % Trials where participants have to make a decision
    end
    
    % Choice trials on map 1
    ix = find(data.scan{run}.RSA.objDiff.choicetrial & data.settings.seq{run}(1,:) <=options.nodes);
    
    % Randomise order
    ix = ix(randperm(length(ix)));
    
    % Pick half of those for distance trials
    if data.session == 3
        data.scan{run}.RSA.objDiff.whichchoice(ix(1:options.nodes/2)) = 'd';   % Distances
        data.scan{run}.RSA.objDiff.whichchoice(ix(options.nodes/2+1:options.nodes)) = 'v';   % Values
    else
        data.scan{run}.RSA.objDiff.whichchoice(ix) = 'd';
    end
    
    % Choice trials on map 2
    ix = find(data.scan{run}.RSA.objDiff.choicetrial & data.settings.seq{run}(1,:) > options.nodes);
    
    % Randomise order
    ix = ix(randperm(length(ix)));
    
    % Pick half of those for distance trials
    if data.session == 3
        data.scan{run}.RSA.objDiff.whichchoice(ix(1:options.nodes/2)) = 'd';   % Distances
        data.scan{run}.RSA.objDiff.whichchoice(ix(options.nodes/2+1:options.nodes)) = 'v';   % Values
    else
        data.scan{run}.RSA.objDiff.whichchoice(ix) = 'd';
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

            %         Pick chosen stimulus
            stimForChoice   = data.settings.seq{run}(1,~strcmp(data.scan{run}.RSA.objDiff.whichchoice,""))';               
            choiceOptions_old = choiceOptions;
        else
            choiceOptions = choiceOptions_old;
            flip = randi(length(choiceOptions),2);
            choiceOptions(flip(:,1),1) = choiceOptions(flip(end:-1:1,1),1);
            choiceOptions(flip(:,2),2) = choiceOptions(flip(end:-1:1,2),2);
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


