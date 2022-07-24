% Sequence inforamtion:
% seq(1,:): stimulus
% seq(2,:): factor
% seq(3,:): map
% seq(4,:): probe trial, 1: distance / 1: value
imagesize = 250;

data.scan{bl}.RSA.objDiff.payout = 0;
data.scan{bl}.when_start     = datestr(now,0);
    
clear theImage dim scaling
for map = 1:2
    for i = 1:length(data.stimuli)
        theImage{map}{i} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',data.context(map,:));
    end
end

data.scan{bl}.RSA.objDiff.cr = zeros(options.scanblocklength,1);
data.scan{bl}.RSA.objDiff.RT = zeros(options.scanblocklength,1);

[xCenter, yCenter] = RectCenter(windowRect);
baseRect = [0 0 2*xCenter 2*xCenter];
centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
    
[X,m] = imread([options.imagePath, 'v.png']); symbol.('v') = ind2rgb(X,m);
[X,m] = imread([options.imagePath, 'd.png']); symbol.('d') = ind2rgb(X,m);
sz = size(symbol.('v')); dim_symb.('v') = sz(1:2);
sz = size(symbol.('d')); dim_symb.('d') = sz(1:2);
        
% Black frame
symbol.('v')(:,1,:) = 0;
symbol.('v')(:,end,:) = 0;
symbol.('v')(1,:,:) = 0;
symbol.('v')(end,1,:) = 0;
symbol.('d')(:,1,:) = 0;
symbol.('d')(:,end,:) = 0;
symbol.('d')(1,:,:) = 0;
symbol.('d')(end,1,:) = 0;

    % Set the text
% Set the text
Screen('TextSize', window, 25);
clear textString;
textString{1} = sprintf('                                  Block %u', bl);
textString{2} = 'Drücken Sie einen Knopf wenn Sie bereit sind zu starten.';
for i = 1:2
    Screen('DrawText',window, textString{i}, xCenter*1/5, yCenter+(i-1)*yCenter/10, white);
end

% Draw the word
% DrawFormattedText(window, textString, 'center', 'center', white);
Screen('Flip', window);     % Flip to the screen
key = [];
if ~buttonbox
    KbPressWait;
else
    while (isempty(key))
        IOPort('Purge',button_port)
        [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
    end
end


if options.scan || options.cbs
    textString = 'Der Scanner wird nun starten. Bitte warten Sie...';
    Screen('TextSize', window, 40);

    data.choice.tStart = Screen('Flip', window);

    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('START SPIKE')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    % Wait for first two scan trigger (after dummies)
    % ======================================================================= %

    drawFixation;
    t1 = Screen('Flip', window);
    
    % wait for five dummy scans to complete
    in_trigger = 0;
    while in_trigger <= data.scanTech.dummy
        trigger = [];
        while (isempty(trigger))
            [trigger,ttrigger] = IOPort('Read', trigger_port);
        end
        in_trigger = in_trigger +1;
        data.scan{bl}.RSA.slice(in_trigger) = GetSecs;
    end
    data.scan{bl}.RSA.taskStart = GetSecs;
else
    drawFixation;
    t1 = Screen('Flip', window);
    WaitSecs(1);
end

% Update sequence
data.scan{bl}.RSA.seq       = data.settings.seq{bl}(1,:);

% value map
data.scan{bl}.RSA.map      = (data.scan{bl}.RSA.seq(1,:) > options.nodes)+1;

data.scan{bl}.RSA.seq(data.scan{bl}.RSA.seq > options.nodes) = data.scan{bl}.RSA.seq(data.scan{bl}.RSA.seq > options.nodes) - options.nodes;

% factors for later factorial analysis
data.scan{bl}.RSA.factors   = data.settings.seq{bl}(2,:);
data.scan{bl}.RSA.start           = datestr(now,0);
data.scan{bl}.RSA.choice         = zeros(1,options.scanblocklength);
data.scan{bl}.RSA.cr             = zeros(1,options.scanblocklength);
data.scan{bl}.RSA.RT             = zeros(1,options.scanblocklength);
data.scan{bl}.RSA.objDiff.payout    = [];

%%
for trial = 1:options.scanblocklength
    checkEscape;
    [cx, xy] = RectCenter(windowRect);
    
    % Set the color of our square to full red. Color is defined by red green
    % and blue components (RGB). So we have three numbers which
    % define our RGB values. The maximum number for each is 1 and the minimum
    % 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
    % 1]. Play around with these numbers and see the result.
    rectColor = data.context(data.scan{bl}.RSA.map(trial),:);
    
    % Draw the square to the screen. For information on the command used in
    % this line see Screen FillRect?
    Screen('FillRect', window, rectColor, centeredRect);
        
    drawFixation;

    % Flip to the screen
    data.scan{bl}.contextOn(trial) = Screen('Flip', window);
    
    % Wait for 1000 ms
    WaitSecs(0.75);
    
    % Draw the square to the screen. For information on the command used in
    % this line see Screen FillRect?
    Screen('FillRect', window, rectColor, centeredRect);
    
    % Make sure the images are appropriately scaled. They should always
    % have a h1ight of 200
    dim = size(theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.seq(trial)});     
    scaling = dim(1)/imagesize;
    
    % Draw the two options to the screen.
    tex = Screen('MakeTexture', window, theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.seq(trial)});
    Screen('DrawTexture', window, tex, [], CenterRectOnPoint([0 0 dim(2)/scaling dim(1)/scaling], cx, xy));
    
    % Flip to the screen
    data.scan{bl}.RSA.stimOn(trial) = Screen('Flip', window);
    
    WaitSecs(2);
    
    Screen('FillRect', window, [0.5 0.5 0.5], centeredRect);
    
    drawFixation;
    data.scan{bl}.RSA.stimOff(trial) = Screen('Flip', window);
    WaitSecs(data.scan{bl}.RSA.jitter(trial));
    
    if ~strcmp(data.scan{bl}.RSA.objDiff.whichchoice(trial),"")
        Screen('FillRect', window, rectColor, centeredRect);
    
        % Display two choice opjects
        clear o
        o{1}  = theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.objDiff.choiceOptions(trial,1)};
        o{2}  = theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.objDiff.choiceOptions(trial,2)};
        
        % Stimulus dimensions
        opt_dim{1} = size(theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.objDiff.choiceOptions(trial,1)});     
        scaling1 = opt_dim{1}(1)/imagesize;
        opt_dim{2} = size(theImage{data.scan{bl}.RSA.map(trial)}{data.scan{bl}.RSA.objDiff.choiceOptions(trial,2)});     
        scaling2 = opt_dim{2}(1)/imagesize;
    
        [cx, xy] = RectCenter(windowRect);
        
        tex1 = Screen('MakeTexture', window, squeeze(o{1}));
        Screen('DrawTexture', window, tex1, [], CenterRectOnPoint([0 0 opt_dim{1}(2)/scaling1 opt_dim{1}(1)/scaling1], cx*0.7, xy));
        tex2 = Screen('MakeTexture', window, squeeze(o{2})); % upside-down
        Screen('DrawTexture', window, tex2, [], CenterRectOnPoint([0 0 opt_dim{2}(2)/scaling2 opt_dim{2}(1)/scaling2], cx*1.3, xy));
        
        % Display symbol indicating which choice
        
        tex_symb = Screen('MakeTexture',window,symbol.(data.scan{bl}.RSA.objDiff.whichchoice(trial)));
        Screen('DrawTexture', window, tex_symb, [], CenterRectOnPoint([0 0 dim_symb.(data.scan{bl}.RSA.objDiff.whichchoice(trial))], cx, xy*0.3));
        
        data.scan{bl}.RSA.objDiff.stimOn(trial) = Screen('Flip', window);
        
        key = [];
        if buttonbox
            key = [];
            while (isempty(key))
                IOPort('Purge',button_port)
                [key,tkey] = IOPort('Read', button_port, 1, 1);
            end
        else
            [tkey,key, deltaSecs] = KbWait([],3);
            while any(find(key)~=leftKey) && any(find(key)~=rightKey)
                [tkey,key, deltaSecs] = KbWait([],3);                      
            end
            key = find(key);
        end
        data.scan{bl}.RSA.objDiff.tKeyPress(trial) = tkey;
        

        if key==leftKey
            data.scan{bl}.RSA.objDiff.choice(trial)         = 1;
            
            data.scan{bl}.RSA.objDiff.distRelCh(trial)      = data.scan{bl}.RSA.objDiff.relValueDist(trial,1);     % Chosen distance relevant map
            data.scan{bl}.RSA.objDiff.distRelUnch(trial)    = data.scan{bl}.RSA.objDiff.relValueDist(trial,2);     % Chosen distance irrelevant map
            data.scan{bl}.RSA.objDiff.distIrrelCh(trial)      = data.scan{bl}.RSA.objDiff.irrelValueDist(trial,1);     % Unhosen distance relevant map
            data.scan{bl}.RSA.objDiff.distIrrelUnch(trial)    = data.scan{bl}.RSA.objDiff.irrelValueDist(trial,2);     % Uncosen distance irrelevant map
            data.scan{bl}.RSA.objDiff.spatialCh(trial)      = data.scan{bl}.RSA.objDiff.('d')(trial,1);     % Chosen distance spatial
            data.scan{bl}.RSA.objDiff.spatialUnch(trial)    = data.scan{bl}.RSA.objDiff.('d')(trial,2);     % Uncosen distance spatial
            
         elseif  key==rightKey
            data.scan{bl}.RSA.objDiff.choice(trial)    = 2;
            
            data.scan{bl}.RSA.objDiff.distRelCh(trial)      = data.scan{bl}.RSA.objDiff.relValueDist(trial,2);     % Chosen distance relevant map
            data.scan{bl}.RSA.objDiff.distRelUnch(trial)    = data.scan{bl}.RSA.objDiff.relValueDist(trial,1);     % Chosen distance irrelevant map
            data.scan{bl}.RSA.objDiff.distIrrelCh(trial)    = data.scan{bl}.RSA.objDiff.irrelValueDist(trial,2);     % Unhosen distance relevant map
            data.scan{bl}.RSA.objDiff.distIrrelUnch(trial)  = data.scan{bl}.RSA.objDiff.irrelValueDist(trial,1);     % Uncosen distance irrelevant map
            data.scan{bl}.RSA.objDiff.spatialCh(trial)      = data.scan{bl}.RSA.objDiff.('d')(trial,2);     % Chosen distance spatial
            data.scan{bl}.RSA.objDiff.spatialUnch(trial)    = data.scan{bl}.RSA.objDiff.('d')(trial,1);     % Uncosen distance spatial
            
         else
            data.scan{bl}.RSA.objDiff.choice(trial)    = key;
        end
        
        if strcmp(data.scan{bl}.RSA.objDiff.whichchoice(trial),"v") && data.scan{bl}.RSA.objDiff.distRelCh(trial) <= data.scan{bl}.RSA.objDiff.distRelUnch(trial)
            data.scan{bl}.RSA.objDiff.cr(trial) = 1;
        elseif strcmp(data.scan{bl}.RSA.objDiff.whichchoice(trial),"d") && data.scan{bl}.RSA.objDiff.spatialCh(trial) <= data.scan{bl}.RSA.objDiff.spatialUnch(trial)
            data.scan{bl}.RSA.objDiff.cr(trial) = 1;
        else
            data.scan{bl}.RSA.objDiff.cr(trial) = -1;
        end
        
        data.scan{bl}.RSA.objDiff.RT(trial)        = data.scan{bl}.RSA.objDiff.tKeyPress(trial) - data.scan{bl}.RSA.objDiff.stimOn(trial);
        
        Screen('FillRect', window, [0.5 0.5 0.5], centeredRect);
        
        drawFixation;
        data.scan{bl}.RSA.objDiff.stimOff(trial) = Screen('Flip', window);
        
        disp([trial data.scan{bl}.RSA.objDiff.RT(trial) key data.scan{bl}.RSA.objDiff.cr(trial)]);
        WaitSecs(data.scan{bl}.RSA.objDiff.jitter(trial));
    end
    
    saveData;
    checkEscape;
    Screen('Close') 
    if subjNo == 115
        data.scan{bl}.RSA.objDiff.payout    = max(0,sum(data.scan{bl}.RSA.objDiff.cr))/10;
    else
        data.scan{bl}.RSA.objDiff.payout    = max(0,sum(data.scan{bl}.RSA.objDiff.cr==1))/10;
    end
end

data.scan{bl}.RSA.objDiff.correct   = sum(data.scan{bl}.RSA.objDiff.cr==1);
data.scan{bl}.RSA.objDiff.incorrect = sum(data.scan{bl}.RSA.objDiff.cr==-1);
data.scan{bl}.RSA.objDiff.meanRT    = [mean(data.scan{bl}.RSA.objDiff.RT(data.scan{bl}.RSA.objDiff.choice ~= 0)), std(data.scan{bl}.RSA.objDiff.RT(data.scan{bl}.RSA.objDiff.choice ~= 0))];

Screen('FillRect', window, [0.5 0.5 0.5], centeredRect);
Screen('Flip', window);

WaitSecs(12);   % End of block
% if options.scan, waitslice(data.scanTech.scanPort,data.scanTech.total_slices + 4*data.scanTech.nslice * data.scanTech.TR); end  % wait for total_slices and 4 more volumes

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('STOP SPIKE')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

data.scan{bl}.when_end     = datestr(now,0);