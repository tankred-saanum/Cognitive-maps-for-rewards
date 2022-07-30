% Training phase
% =========================================================================
% Here we load 4 images from file - data.mapSize objects, 2 contexts, 2 orientations
data.choice.when_start     = datestr(now,0);
imagesize = 250;

clear theImage scaling
clear dim
for map = 1:2
    for i = 1:length(data.stimuli)
        theImage{map}{i} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',data.context(map,:));
    end
end  

data.choice.payout    = [];

% Set the text
% textString = 'Bitte entscheiden Sie sich zwischen den Monstern und sammeln Sie so viele Punkte wie möglich.\n\nKonzentrieren Sie sich gut, Sie haben nur diesen einen Block um die zwei Punkteverteilungen zu lernen.\n\nDrücken Sie eine Taste wenn Sie bereit sind zu starten.';

% textString = 'Bitte entscheiden Sie sich zwischen den Monstern und sammeln Sie so viele Punkte wie möglich.\n\nKonzentrieren Sie sich gut, Sie haben nur diesen einen Block um die zwei Punkteverteilungen zu lernen.';Screen('TextSize', window, 15);
clear textString
Screen('TextSize', window, 20);
textString{1} = 'Sie können nun zwischen zwei Monstern wählen. Je nachdem, für welches';
textString{2} = 'Monster Sie sich entscheiden, bekommen Sie unterschiedlich viele Punkte.';
textString{3} = 'Wie viele Punkte Sie für ein Monster bekommen hängt von der Hintergrundfarbe';
textString{4} = 'und von der Position des Monsters im Raum ab. Monster, die nahe beieinander';
textString{5} = 'liegen, bringen auch ähnlich viele Punkte. Versuchen Sie so viele Punkte wie';
textString{6} = 'möglich zu sammeln. Konzentrieren Sie sich gut, Sie haben nur diesen einen';
textString{7} = 'Block um die Punkteverteilungen zu lernen.';
textString{8} = '';
textString{9} = 'Drücken Sie eine Taste wenn Sie bereit sind zu starten.';
for i = 1:9
    Screen('DrawText',window, textString{i}, xCenter/10, yCenter-yCenter/2+i*yCenter/10, white);
end
Screen('Flip', window);     % Flip screen

% Draw the word
% DrawFormattedText(window, textString, 'center', 'center', white);
% Screen('Flip', window);     % Flip to the screen
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

% The experimenter starts the script by pressing 'A' then 'C'
% [~,key] = KbPressWait;
% while find(key)~=KbName('A')
%     disp(find(key))
%     [~,key] = KbPressWait;
% end
% while find(key)~=KbName('C')
%     [~,key] = KbPressWait;
% end
% Screen('Flip', window);     % Flip to the screen

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
        data.choice.slice(in_trigger) = GetSecs;
    end
    data.choice.taskStart = GetSecs;
        
else
    drawFixation;
    t1 = Screen('Flip', window);
    WaitSecs(1);
end

% Initialise variables to store choice behaviour
data.choice.start          = datestr(now,0);
data.choice.decision       = [];
data.choice.chosen_value   = [];
data.choice.unchosen_value = [];
data.choice.chosen_object  = [];
data.choice.unchosen_object = [];
data.choice.cr             = [];
data.choice.RT             = [];

% Generate choice sequence
% randomise order in which maps are learned
if mod(str2double(data.subject),4) > 1 
    order = 0;
else
    order = 1;
end
    
data.choice.map = repmat([2*ones(1,10)-mod(order,2) ones(1,10)+mod(order,2)],1,options.choiceblocklength/20);
for map = 1:2
    for o = 1:2
        % Remove inference objects from choice trials!
        choice_options = 1:options.nodes;
        choice_options(options.inference_objects(map,:)) = [];
        
        % Split choice blocks for each context in half
        option{map}(o,:) = repmat(choice_options,1,options.choiceblocklength/length(choice_options)/2);
        option{map}(o,:) = option{map}(1,randperm(length(option{map}(o,:))));
    end
    while any(diff(option{map}) == 0)
        option{map}(1,:) = option{map}(1,randperm(length(option{map}(1,:))));
    end
    data.choice.options(:,data.choice.map == map) = option{map};
    data.choice.values(1,data.choice.map == map) = data.settings.value(map,option{map}(1,:));
    data.choice.values(2,data.choice.map == map) =  data.settings.value(map,option{map}(2,:));
end

for trial = 1:options.choiceblocklength
    % jitter between 4 and 8 seconds
    
    if trial > 1 && data.choice.map(trial) ~= data.choice.map(trial-1)
        
        % Set the text
        textString = 'Achtung, nun ändert sich die Hintergrundfarbe.';
        Screen('TextSize', window, 20);
        Screen('DrawText',window, textString, xCenter-xCenter/2, yCenter, white);
        
        % Draw the word
%         DrawFormattedText(window, textString, 'center', 'center', white);
        Screen('Flip', window);     % Flip to the screen
    
        % jitter
        WaitSecs(2)   ;  
    end
    
    drawFixation;
    Screen('Flip', window);
    WaitSecs(0.5);
    
    checkEscape;
        
    % Make sure the images are appropriately scaled. They should always
    % have a hight of 200
    dim{1} = size(theImage{data.choice.map(trial)}{data.choice.options(1,trial)});     scaling{1} = dim{1}(1)/imagesize;
    dim{2} = size(theImage{data.choice.map(trial)}{data.choice.options(2,trial)});     scaling{2} = dim{2}(1)/imagesize;
    
    baseRect = [0 0 2*xCenter 2*yCenter];
    centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
    
    % Set the color of our square to full red. Color is defined by red green
    % and blue components (RGB). So we have three numbers which
    % define our RGB values. The maximum number for each is 1 and the minimum
    % 0. So, "full red" is [1 0 0]. "Full green" [0 1 0] and "full blue" [0 0
    % 1]. Play around with these numbers and see the result.
    rectColor = data.context(data.choice.map(trial),:);
    
    % Draw the square to the screen. For information on the command used in
    % this line see Screen FillRect?
    Screen('FillRect', window, rectColor, centeredRect);
    
    % Draw the two options to the screen.
    tex(1) = Screen('MakeTexture', window, theImage{data.choice.map(trial)}{data.choice.options(1,trial)});
    Screen('DrawTexture', window, tex(1), [], CenterRectOnPoint([0 0 dim{1}(2)/scaling{1} dim{1}(1)/scaling{1}], xCenter*0.7, yCenter));
    tex(2) = Screen('MakeTexture', window, theImage{data.choice.map(trial)}{data.choice.options(2,trial)}); % upside-down
    Screen('DrawTexture', window, tex(2), [], CenterRectOnPoint([0 0 dim{2}(2)/scaling{2} dim{2}(1)/scaling{2}], xCenter*1.3, yCenter));
    
    
    % display options DECIDE phase
    data.choice.stimOn(trial) =  Screen('Flip', window);   
    if buttonbox
        key = [];
        while isempty(key) || ( key~=leftKey && key~=rightKey)
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
        
    % Which key did the subject press?
    if key==leftKey
        data.choice.decision(trial)    = 1;
    elseif key==rightKey
        data.choice.decision(trial)    = 2;
    else
        data.choice.decision(trial)    = key;
    end
    choice = data.choice.decision(trial);
    
    data.choice.tKeyPress(trial) = tkey;
    
    data.choice.chosen_object(trial)       = data.choice.options(choice,trial);
    data.choice.unchosen_object(trial)     = data.choice.options(choice + (-1)^(choice+1),trial);
    data.choice.chosen_value(trial)        = data.choice.values(choice,trial);
    data.choice.unchosen_value(trial)      = data.choice.values(choice + (-1)^(choice+1),trial);
    data.choice.cr(trial)                  = data.choice.values(choice,trial) >= data.choice.values(choice + (-1)^(choice+1),trial);
    data.choice.RT(trial)                  = tkey - data.choice.stimOn(trial);        
   
    % Draw choice + outcome
    Screen('FillRect', window, rectColor, centeredRect);
    Screen('DrawTexture', window, tex(choice), [], CenterRectOnPoint([0 0 dim{choice}(2)/scaling{choice} dim{choice}(1)/scaling{choice}], xCenter*(0.1+0.6*choice), yCenter));    
    data.choice.decisionTime(trial) = Screen('Flip', window);
    
    WaitSecs(1.5);
    
    drawFixation;
    Screen('Flip', window);
    
    % jitter
    WaitSecs(data.choice.jitter(trial)-1.5);
    
    % Draw choice + outcome
%     Screen('DrawTexture', window, tex(choice), [], CenterRectOnPoint([0 0 dim{choice}(2)/scaling{choice} dim{choice}(1)/scaling{choice}], cx*(0.4+0.4*choice), xy));    
        
%     c = Screen('MakeTexture', window, coins{data.choice.map(trial)}{min(round(data.choice.chosen_value(trial)/2)+1,50)}); % upside-down
%     Screen('DrawTexture', window, c, [], CenterRectOnPoint([0 0 dim_coins(2)/scaling_coins dim_coins(1)/scaling_coins], cx, xy));    
    
    textString = num2str(floor (data.choice.chosen_value(trial)-0.01));
    %     DrawFormattedText(window, textString, 'center', 'center', white);
    % Set the text
    Screen('TextSize', window, 30);
    Screen('FillRect', window, [0.50 0.5 0.5], centeredRect);
    
    % Draw the word
    Screen('DrawText',window, textString, xCenter - 0.05*xCenter,yCenter- 0.05*yCenter, white);
    
    % Draw the word
    %         DrawFormattedText(window, textString, 'center', 'center', white);
    
    data.choice.feedbackOn(trial) = Screen('Flip', window);     % Flip screen
    
    WaitSecs(2);
    
    drawFixation;
    data.choice.feedbackOff(trial) = Screen('Flip', window);
    
    % jitter
    WaitSecs(data.choice.fb_jitter(trial)-2);      
    saveData
end


data.choice.payout    = sum(data.choice.chosen_value)/1000;

drawFixation;
Screen('Flip', window);

try
    saveData;
catch
    disp('Error')
end

WaitSecs(12);   % End of block


disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('STOP SPIKE')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


data.choice.when_end     = datestr(now,0);