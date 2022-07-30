%% Training phase
% =========================================================================

% load sequence information
load([options.root,'/expt_specs/seq_v',num2str(bl),'_',num2str(num2str(data.subject)),'.mat'])
data.scan{bl} = sq;
    
% Here we load 64 images from file - data.mapSize objects, 2 contexts, 2 orientations
clear theImage
for map = 1:2
    for i = 1:length(data.stimuli)
        theImage{map}{i} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',data.scan{bl}.context(map,:));
    end
    for i = 1:50
        coins{map}{i} = imread([options.imagePath, sprintf('coins/%dcoins.png',i)],'BackgroundColor',data.scan{bl}.context(map,:));
    end
end
dim_coins = size(coins{1}{1}); 

if options.scan
    open_IOport;
    IOport_logic; %defines IO.trig, IO.def, IO.buttons (4 rows)
    buttonbox     = 1;
else
    buttonbox     = 0;
end

if buttonbox && ~options.scan
    open_IOport;
    IOport_logic;
end

% Set the text
textString = sprintf('Start of block %u\n\nPress any key when you are ready to start.',bl);
Screen('TextSize', window, 30);

% Draw the word
DrawFormattedText(window, textString, 'center', 'center', white);
Screen('Flip', window);     % Flip to the screen
key = [];    
if ~buttonbox
    KbPressWait;
else
    key = [];
    while isempty(key) || (key~=leftKey && key~=rightKey)
        IOPort('Purge',button_port)
        [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
    end
end

textString = 'The scanner will now start...';
Screen('TextSize', window, 40);

data.scan{bl}.tStart = Screen('Flip', window);

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('START SPIKE')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


% Wait for first two scan trigger (after dummies)
% ======================================================================= %
if options.scan
    drawFixation;
    t1 = Screen('Flip', window);
    
    %-wait for scanner to start (fmrib: one trigger per volume)
    byte_in = 0;
    while ~bitget(byte_in,IO.trigBit) %(byte_in~=IO.trig)
        byte_in=io64(ioObj,address);
        WaitSecs(0.005);
    end
    SLICE.trig1 = GetSecs;
    data.scan{bl}.tTrig1 = SLICE.trig1;
    
    %-wait for trigger to be reset again (80ms)
    %byte_in = 0;
    while bitget(byte_in,IO.trigBit) %(byte_in~=IO.def)
        byte_in=io64(ioObj,address);
        WaitSecs(0.005);
    end
    
    %-apparently at FMRIB no triggers for 5 dummy vols send, so just
    % wait for one more then start
    volumeCtt = 1;
    while volumeCtt<5%SLICE.dummy
        byte_in=io64(ioObj,address);
        if bitget(byte_in,IO.trigBit) %byte_in==IO.trig
            volumeCtt = volumeCtt+1;
        end
        WaitSecs(0.005);
    end
    SLICE.trig2	= GetSecs;
    data.scan{bl}.tTrig2 = SLICE.trig2;
else
    drawFixation;
    t1 = Screen('Flip', window);
    WaitSecs(1);
end

% Initialise variables to store choice behaviour
data.scan{bl}.choice.start          = datestr(now,0);
data.scan{bl}.choice.decision       = [];
data.scan{bl}.choice.chosen_value   = [];
data.scan{bl}.choice.unchosen_value = [];
data.scan{bl}.choice.chosen_object  = [];
data.scan{bl}.choice.unchosen_object = [];
data.scan{bl}.choice.cr             = [];
data.scan{bl}.choice.RT             = [];

% Generate choice sequence
% randomise order in which maps are learned
order = rand(1)>0.5;
data.scan{bl}.choice.map = repmat([2*ones(1,options.choiceblocklength/4)-mod(order,2) ones(1,options.choiceblocklength/4)+mod(order,2)],1,2);
for map = 1:2
    for o = 1:2
        option{map}(o,:) = repmat(1:options.nodes,1,options.choiceblocklength/options.nodes/2);
        option{map}(o,:) = option{map}(1,randperm(length(option{map}(o,:))));
    end
    while any(diff(option{map}) == 0)
        option{map}(1,:) = option{map}(1,randperm(length(option{map}(1,:))));
    end
    data.scan{bl}.choice.options(:,data.scan{bl}.choice.map == map) = option{map};
    data.scan{bl}.choice.values(1,data.scan{bl}.choice.map == map) = data.scan{bl}.valuedistr(map,option{map}(1,:));
    data.scan{bl}.choice.values(2,data.scan{bl}.choice.map == map) = data.scan{bl}.valuedistr(map,option{map}(2,:));
end

for trial = 1:options.choiceblocklength
    checkEscape;
    
    [cx, xy] = RectCenter(windowRect);
    
    % Make sure the images are appropriately scaled. They should always
    % have a hight of 200
    dim{1} = size(theImage{data.scan{bl}.choice.map(trial)}{data.scan{bl}.choice.options(1,trial)});     scaling{1} = dim{1}(1)/200;
    dim{2} = size(theImage{data.scan{bl}.choice.map(trial)}{data.scan{bl}.choice.options(2,trial)});     scaling{2} = dim{2}(1)/200;
    
    [xCenter, yCenter] = RectCenter(windowRect);
    baseRect = [0 0 2*cx 2*xy];
    centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
    
    % Set the color of our square to full red. Color is defined by red green
    % and blue components (RGB). So we have three numbers which
    % define our RGB values. The maximum number for each is 1 and the minimum
    % 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
    % 1]. Play around with these numbers and see the result.
    rectColor = data.scan{bl}.context(data.scan{bl}.choice.map(trial),:);
    
    % Draw the square to the screen. For information on the command used in
    % this line see Screen FillRect?
    Screen('FillRect', window, rectColor, centeredRect);
    
    % Draw the two options to the screen.
    tex(1) = Screen('MakeTexture', window, theImage{data.scan{bl}.choice.map(trial)}{data.scan{bl}.choice.options(1,trial)});
    Screen('DrawTexture', window, tex(1), [], CenterRectOnPoint([0 0 dim{1}(2)/scaling{1} dim{1}(1)/scaling{1}], cx*0.8, xy));
    tex(2) = Screen('MakeTexture', window, theImage{data.scan{bl}.choice.map(trial)}{data.scan{bl}.choice.options(2,trial)}); % upside-down
    Screen('DrawTexture', window, tex(2), [], CenterRectOnPoint([0 0 dim{2}(2)/scaling{2} dim{2}(1)/scaling{2}], cx*1.2, xy));
    
    % display options DECIDE phase
    data.scan{bl}.choice.stimOn(trial) =  Screen('Flip', window);   
    
    key = [];
    if buttonbox
        [tkey,key] = IOWaitButton(IO,inf,ioObj,address);
    else
        [tkey,key, deltaSecs] = KbPressWait;
        while find(key)~=KbName('LeftArrow') && find(key)~=KbName('RightArrow') 
            [tkey,key] = KbPressWait;
        end
    end
    
    % Which key did the subject press?
    if find(key)==KbName('LeftArrow')
        data.scan{bl}.choice.decision(trial)    = 1;
    elseif find(key)==KbName('RightArrow')
        data.scan{bl}.choice.decision(trial)    = 2;
    else
        data.scan{bl}.choice.decision(trial)    = find(key);
    end
    choice = data.scan{bl}.choice.decision(trial);
    
    data.scan{bl}.choice.tKeyPress(trial) = tkey;
    
    data.scan{bl}.choice.chosen_object(trial)   = data.scan{bl}.choice.options(choice,trial);
    data.scan{bl}.choice.unchosen_object(trial) = data.scan{bl}.choice.options(choice + (-1)^(choice+1),trial);
    data.scan{bl}.choice.chosen_value(trial)   = data.scan{bl}.choice.values(choice,trial);
    data.scan{bl}.choice.unchosen_value(trial) = data.scan{bl}.choice.values(choice + (-1)^(choice+1),trial);
    data.scan{bl}.choice.cr(trial)             = data.scan{bl}.choice.values(choice,trial) <= data.scan{bl}.choice.values(choice + (-1)^(choice+1),trial);
    data.scan{bl}.choice.RT(trial)             = tkey - data.scan{bl}.choice.stimOn(trial);        
   
    % Draw choice + outcome
    Screen('DrawTexture', window, tex(choice), [], CenterRectOnPoint([0 0 dim{choice}(2)/scaling{choice} dim{choice}(1)/scaling{choice}], cx*(0.4+0.4*choice), xy));    
    Screen('Flip', window);
    
    % jitter between 4 and 8 seconds
    WaitSecs(randi([30 60])/10)   ;    
    
    % Draw choice + outcome
    Screen('DrawTexture', window, tex(choice), [], CenterRectOnPoint([0 0 dim{choice}(2)/scaling{choice} dim{choice}(1)/scaling{choice}], cx*(0.4+0.4*choice), xy));    
        
    scaling_coins = 3;
    c = Screen('MakeTexture', window, coins{data.scan{bl}.choice.map(trial)}{round(data.scan{bl}.choice.chosen_value(trial)/2)}); % upside-down
    Screen('DrawTexture', window, c, [], CenterRectOnPoint([0 0 dim_coins(2)/scaling_coins dim_coins(1)/scaling_coins], cx, xy+xy/2));    
    
    Screen('Flip', window);
    
    % jitter between 4 and 8 seconds
    WaitSecs(randi([30 60])/10);
    
end

data.scan{bl}.objDiff.RT(trial)        = data.scan{bl}.objDiff.tKeyPress(trial) - data.scan{bl}.objDiff.stimOn(trial);

drawFixation;
Screen('Flip', window);
WaitSecs(data.scan{bl}.jitter(trial));



choice = yield choiceTrial([stimuli[0][choiceTask['choice_option0'][0][c]], stimuli[0][choiceTask['choice_option1'][0][c]]])
choiceTask['RT'].append(choice[1].time - choice[0].time)

if find(key)==leftKey
    data.scan{bl}.choice.decision(trial)        = 1;
    data.scan{bl}.choice.chosen_value(trial)    = data.scan{bl}.choice.chosen_value(trial);
    data.scan{bl}.choice.unchosen_value(trial)  = 0;
    
    choiceTask['chosen_value'].append(choiceTask['choice_value0'][0][c])
    choiceTask['unchosen_value'].append(choiceTask['choice_value1'][0][c])
    
    if choiceTask['choice_value0'][0][c] > choiceTask['choice_value1'][0][c]:
        choiceTask['cr'].append(1)
    else:
        choiceTask['cr'].append(0)
        quad1.remove()
        pos = 0.3
    else:
        choiceTask['decision'].append(1)
        choiceTask['chosen_value'].append(choiceTask['choice_value1'][0][c])
        choiceTask['unchosen_value'].append(choiceTask['choice_value0'][0][c])
        
        if choiceTask['choice_value1'][0][c] > choiceTask['choice_value0'][0][c]:
            choiceTask['cr'].append(1)
        else:
            choiceTask['cr'].append(0)
            quad0.remove()
            pos = 0.7
            
            progressBarOn = True
            presentProgressBar(choiceTask['chosen_value'][c],pos)
            yield viztask.waitTime(2)
            
            quad0.remove()
            quad1.remove()
            myProgressBar.remove()
            yield presentFixationCross(0.5)
            
            saveData()
            
            choiceTask['totalRew'] = np.sum(choiceTask['chosen_value'])
            totalRew += choiceTask['totalRew']
            
            choiceTask['end'] = time.clock()
            
        end
        
        
        data.scan{bl}.when_start             = datestr(now,0);
        data.scan{bl}.objDiff.choice         = zeros(1,options.scanblocklength);
        data.scan{bl}.objDiff.cr             = zeros(1,options.scanblocklength);
        data.scan{bl}.objDiff.RT             = zeros(1,options.scanblocklength);
        
        for trial = 1:options.scanblocklength
            checkEscape;
            
            stimTrial = data.scan{bl}.seq(1,trial);
            if stimTrial == data.mapSize
                map = 1; stimulus = data.mapSize;
            elseif stimTrial == data.conditions
                map = 2; stimulus = data.mapSize;
            else
                map = floor(stimTrial/data.mapSize)+1;
                stimulus = mod(stimTrial,data.mapSize);
            end
            
            data.scan{bl}.map(trial) = map;
            
            % Present the context for 500 ms
            % This is just a small box on the screen
            
            % % Make the image into a texture
            imageTexture = Screen('MakeTexture', window, background{map});
            
            % Draw the image to the screen, unless otherwise specified PTB will draw
            % the texture full size in the center of the screen.
            Screen('DrawTexture', window, imageTexture, [], [], 0);
            
            % Flip to the screen
            data.scan{bl}.contextOn(trial) = Screen('Flip', window);
            
            % Wait for 1500 ms
            WaitSecs(1);
            
            % Then present the image for 1 sec
            % % Make the image into a texture
            imageTexture1 = Screen('MakeTexture', window, background{map});
            imageTexture2 = Screen('MakeTexture', window, theImage{stimulus});
            
            % Draw the image to the screen, unless otherwise specified PTB will draw
            % the texture full size in the center of the screen.
            Screen('DrawTexture', window, imageTexture1, [], [], 0);
            Screen('DrawTexture', window, imageTexture2, [], [], 0);
            
            % Flip to the screen
            data.scan{bl}.stimOn(trial) = Screen('Flip', window);
            
            WaitSecs(2);
            
            drawFixation;
            Screen('Flip', window);
            WaitSecs(data.scan{bl}.jitter(trial));
            
            if data.scan{bl}.seq(3,trial) ~= 0
                % Pick two random objects
                % Needs to be replaced such that distances are orthogonalised!
                if data.day ~= 0
                    o(1,:,:,:)= theImage{data.scan{bl}.objDiff.stim(trial,1)}{map}{data.scan{bl}.objDiff.orient(trial,1)}(1:199,1:199,:);
                    o(2,:,:,:) = theImage{data.scan{bl}.objDiff.stim(trial,2)}{map}{data.scan{bl}.objDiff.orient(trial,2)}(1:199,1:199,:);
                else
                    data.scan{bl}.objDiff.stimOrder(trial,:) = randperm(3);
                    clear o
                    o(data.scan{bl}.objDiff.stimOrder(trial,:) == 1,:,:,:) = theImage{stimulus}{map}{orientation}(1:199,1:199,:);
                    o(data.scan{bl}.objDiff.stimOrder(trial,:) == 2,:,:,:) = theImage{data.scan{bl}.objDiff.stim(trial,1)}{map}{data.scan{bl}.objDiff.orient(trial,1)}(1:199,1:199,:);
                    o(data.scan{bl}.objDiff.stimOrder(trial,:) == 3,:,:,:) = theImage{data.scan{bl}.objDiff.stim(trial,2)}{map}{data.scan{bl}.objDiff.orient(trial,2)}(1:199,1:199,:);
                end
                
                [cx, xy] = RectCenter(windowRect);
                
                Screen('DrawTexture', window, imageTexture1, [], [], 0);
                
                if data.day == 0
                    tex1 = Screen('MakeTexture', window, squeeze(o(1,:,:,:)));
                    Screen('DrawTexture', window, tex1, [], CenterRectOnPoint([0 0 200 200], cx*0.7, xy));
                    tex2 = Screen('MakeTexture', window, squeeze(o(2,:,:,:))); % upside-down
                    Screen('DrawTexture', window, tex2, [], CenterRectOnPoint([0 0 200 200], cx*1.3, xy));
                    tex3 = Screen('MakeTexture', window, squeeze(o(3,:,:,:))); % upside-down
                    Screen('DrawTexture', window, tex3, [], CenterRectOnPoint([0 0 200 200], cx*1.0, xy));
                else
                    tex1 = Screen('MakeTexture', window, squeeze(o(1,:,:,:)));
                    Screen('DrawTexture', window, tex1, [], CenterRectOnPoint([0 0 200 200], cx*0.8, xy));
                    tex2 = Screen('MakeTexture', window, squeeze(o(2,:,:,:))); % upside-down
                    Screen('DrawTexture', window, tex2, [], CenterRectOnPoint([0 0 200 200], cx*1.2, xy));
                end
                
                data.scan{bl}.objDiff.stimOn(trial) = Screen('Flip', window);
                
                if buttonbox
                    [tkey,key] = IOWaitButton(IO,inf,ioObj,address);
                else
                    [tkey,key, deltaSecs] = KbWait([],3);
                end
                data.scan{bl}.objDiff.tKeyPress(trial) = tkey;
                
                if data.day == 0 % Pick the correct of the three objects
                    if find(key)==leftKey
                        data.scan{bl}.objDiff.choice(trial)    = 1;
                    elseif find(key)==centreKey % careful, third object is in the center!
                        data.scan{bl}.objDiff.choice(trial)    = 3;
                    elseif  find(key)==rightKey
                        data.scan{bl}.objDiff.choice(trial)    = 2;
                    else
                        data.scan{bl}.objDiff.choice(trial)    = find(key);
                    end
                    if find(key)==leftKey && data.scan{bl}.objDiff.stimOrder(trial,1) == 1 || ...
                            find(key)==centreKey && data.scan{bl}.objDiff.stimOrder(trial,1) == 2 || ...
                            find(key)==rightKey && data.scan{bl}.objDiff.stimOrder(trial,1) == 3
                        data.scan{bl}.objDiff.cr(trial) = 1;
                    else
                        data.scan{bl}.objDiff.cr(trial) = -1;
                    end
                else
                    if find(key)==leftKey
                        data.scan{bl}.objDiff.choice(trial)    = -1;
                        
                        data.scan{bl}.objDiff.distRelCh(trial) = data.map{map,map}(data.scan{bl}.objDiff.stim(trial,1), stimulus);
                        data.scan{bl}.objDiff.distRelUnch(trial) = data.map{map,map}(data.scan{bl}.objDiff.stim(trial,2), stimulus);
                        
                        data.scan{bl}.objDiff.distIrrelCh(trial) = data.map{map,mod(map,2)+1}(data.scan{bl}.objDiff.stim(trial,1), stimulus);
                        data.scan{bl}.objDiff.distIrrelUnch(trial) = data.map{map,mod(map,2)+1}(data.scan{bl}.objDiff.stim(trial,2), stimulus);
                    elseif  find(key)==rightKey
                        data.scan{bl}.objDiff.choice(trial)    = 1;
                        
                        data.scan{bl}.objDiff.distRelCh(trial) = data.map{map,map}(data.scan{bl}.objDiff.stim(trial,2), stimulus);
                        data.scan{bl}.objDiff.distRelUnch(trial) = data.map{map,map}(data.scan{bl}.objDiff.stim(trial,1), stimulus);
                        
                        data.scan{bl}.objDiff.distIrrelCh(trial) = data.map{map,mod(map,2)+1}(data.scan{bl}.objDiff.stim(trial,2), stimulus);
                        data.scan{bl}.objDiff.distIrrelUnch(trial) = data.map{map,mod(map,2)+1}(data.scan{bl}.objDiff.stim(trial,1), stimulus);
                    else
                        data.scan{bl}.objDiff.choice(trial)    = find(key);
                    end
                    if data.scan{bl}.objDiff.distRelCh(trial) <= data.scan{bl}.objDiff.distRelUnch(trial)
                        data.scan{bl}.objDiff.cr(trial) = 1;
                    else
                        data.scan{bl}.objDiff.cr(trial) = -1;
                    end
                end
                
                data.scan{bl}.objDiff.RT(trial)        = data.scan{bl}.objDiff.tKeyPress(trial) - data.scan{bl}.objDiff.stimOn(trial);
                
                drawFixation;
                Screen('Flip', window);
                WaitSecs(data.scan{bl}.jitter(trial));
            end
            disp([trial data.scan{1}.objDiff.RT(trial) find(key) data.scan{bl}.objDiff.cr(trial)]);
            
            save ([options.saveDataPath,'/data_',num2str(data.subjNo),'_',num2str(data.session),'.mat'],'data');
            checkEscape;
        end
        
        data.scan{bl}.objDiff.correct   = sum(data.scan{bl}.objDiff.cr==1);
        data.scan{bl}.objDiff.incorrect = sum(data.scan{bl}.objDiff.cr==-1);
        data.scan{bl}.objDiff.payout    = max(0,sum(data.scan{bl}.objDiff.cr))/10;
        data.scan{bl}.objDiff.meanRT    = [mean(data.scan{bl}.objDiff.RT(data.scan{bl}.objDiff.choice ~= 0)), std(data.scan{bl}.objDiff.RT(data.scan{bl}.objDiff.choice ~= 0))];
        
        Screen('Flip', window);
        
        % if options.scan, waitslice(data.scanTech.scanPort,data.scanTech.total_slices + 4*data.scanTech.nslice * data.scanTech.TR); end  % wait for total_slices and 4 more volumes
        
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        disp('STOP SPIKE')
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
        try
            save ([options.saveDataPath,'/data_',num2str(data.subjNo),'_',num2str(data.session),'.mat'],'data');
        catch
            disp('Error')
        end