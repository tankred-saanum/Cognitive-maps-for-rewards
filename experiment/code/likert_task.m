% Collect likability ratings
clear theImage
for i = 1:length(data.stimuli)
    theImage{i} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',[0.5 0.5 0.5]);
end

% Define variables
width           = 2;
scalaPosition   = 0.8;
lineLength      = 10;
center          = round([windowRect(3) windowRect(4)]/2);    
scalaLength     = 0.9;
dotSizePix      = 30;
dotColor        = [1 0 0];
    
HideCursor;    
    
Screen('TextSize', window, 50);
textString = sprintf('Aufgabe 1\n\n\nBitte drücken Sie eine Taste um zu starten.');
DrawFormattedText(window, textString, 'center', 'center', white);
Screen('Flip', window);
KbPressWait

stimPres = randperm(length(data.stimuli));
for objects = 1:length(stimPres)
    
    data.likert.(['trial_',num2str(objects-1)]).stim        = stimPres(objects);
    data.likert.(['trial_',num2str(objects-1)]).whichPNG    = data.stimuli(stimPres(objects));
    data.likert.(['trial_',num2str(objects-1)]).position    = 0.5;
    
    [cx, xy] = RectCenter(windowRect);
    
    drawFixation;
    Screen('Flip', window);

    WaitSecs(1);
    
    data.likert.(['trial_',num2str(objects-1)]).time = Screen('Flip', window);
    SetMouse(windowRect(3)/2, windowRect(4)*scalaPosition, window);   
    [x, y, buttons] = GetMouse(window);
    HideCursor;    
        
    while sum(buttons) == 0
        
        % Draw the two options to the screen.
        tex = Screen('MakeTexture', window, theImage{stimPres(objects)});

        % Make sure the images are appropriately scaled. They should always
        % have a h1ight of 200
        dim = size(theImage{stimPres(objects)});     scaling = dim(1)/400;

        Screen('DrawTexture', window, tex, [], CenterRectOnPoint([0 0 dim(2)/scaling dim(1)/scaling], cx, xy));

        endPoints = {sprintf('Überhaupt\n    nicht'), sprintf('Sehr\n gut')};

        Screen('TextSize', window, 50);
        % Drawing the question as text
        DrawFormattedText(window, 'Wie gut gefällt Ihnen dieses Monster?', 'center', windowRect(4)*0.2, white);

        midTick    = [center(1) windowRect(4)*scalaPosition - lineLength center(1) windowRect(4)*scalaPosition  + lineLength];
        leftTick   = [windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition - lineLength windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition  + lineLength];
        rightTick  = [windowRect(3)*scalaLength windowRect(4)*scalaPosition - lineLength windowRect(3)*scalaLength windowRect(4)*scalaPosition  + lineLength];
        horzLine   = [windowRect(3)*scalaLength windowRect(4)*scalaPosition windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition];
        textBounds = [Screen('TextBounds', window, endPoints{1}); Screen('TextBounds', window, endPoints{2})];

        % Drawing the scala
        Screen('DrawLine', window, [1 1 1], midTick(1), midTick(2), midTick(3), midTick(4), width);         % Mid tick
        Screen('DrawLine', window, [1 1 1], leftTick(1), leftTick(2), leftTick(3), leftTick(4), width);     % Left tick
        Screen('DrawLine', window, [1 1 1], rightTick(1), rightTick(2), rightTick(3), rightTick(4), width); % Right tick
        Screen('DrawLine', window, [1 1 1], horzLine(1), horzLine(2), horzLine(3), horzLine(4), width);     % Horizontal line

        % Drawing the end points of the scala as text
        Screen('TextSize', window, 30);
        DrawFormattedText(window, endPoints{1}, leftTick(1, 1)-50,  windowRect(4)*scalaPosition+40, [],[],[],[],[],[],[]); % Left point
        DrawFormattedText(window, endPoints{2}, rightTick(1, 1) - textBounds(2, 3)/2+50,  windowRect(4)*scalaPosition+40, [],[],[],[],[],[],[]); % Right point
        
        % Get the current position of the mouse
        [x, y, buttons] = GetMouse(window);
        
        % We clamp the values at the maximum values of the screen in X and Y
        % incase people have two monitors connected. This is all we want to
        % show for this basic demo.
        x = max(min(x, windowRect(3)*scalaLength), round(windowRect(3)*(1-scalaLength)));
        
        Screen('DrawDots', window, [x windowRect(4)*scalaPosition], dotSizePix, dotColor, [], 2);
        
        % Flip to the screen
        Screen('Flip', window);
    end
    
    data.likert.(['trial_',num2str(objects-1)]).position(end+1) = (x - windowRect(3) * (1-scalaLength)) / (windowRect(3)*scalaLength - round(windowRect(3)*(1-scalaLength)));
                
    % Flip to the screen
    data.likert.(['trial_',num2str(objects-1)]).time(end+1) = Screen('Flip', window);
    data.likert.(['trial_',num2str(objects-1)]).RT       = data.likert.(['trial_',num2str(objects-1)]).time(end) - data.likert.(['trial_',num2str(objects-1)]).time(1);
    
    save ([options.dataPath,'/data_',data.subject,'_',num2str(session),'_',initials,'_postscan.mat'],'data');
end