% Collect likability ratings
clear theImage
for i = 1:length(data.stimuli)
    theImage{i} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',data.context(1,:));
    theImage{i+length(data.stimuli)} = imread([options.imagePath, sprintf('obj%02d.png',data.stimuli(i))],'BackgroundColor',data.context(2,:));
end

for c = 1:2
    coin{c} = imread([options.imagePath, 'coins/1coins.png'],'BackgroundColor',data.context(c,:));
    coin{c} = coin{c}(10:120,20:110,:);
end

Screen('TextSize', window, 50);
textString = sprintf('Aufgabe 2\n\nBitte drücken Sie eine Taste um zu starten.');
DrawFormattedText(window, textString, 'center', 'center', white);
Screen('Flip', window);
KbPressWait

% Define variables
width           = 2;
scalaPosition   = 0.8;
lineLength      = 10;
center          = round([windowRect(3) windowRect(4)]/2);    
scalaLength     = 0.9;
dotSizePix      = 15;
dotColor        = [1 0 0];
    
HideCursor;    
    
leftlim = windowRect(3) * (1-scalaLength);
rightlim = windowRect(3) * scalaLength;

stimPres = randperm(length(data.stimuli)*2);
for objects = 1:length(data.stimuli)*2
    
    [cx, xy] = RectCenter(windowRect);
    data.value_rating.(['trial_',num2str(objects-1)]).stim        = stimPres(objects);
    if stimPres(objects) <= length(data.stimuli)
        data.value_rating.(['trial_',num2str(objects-1)]).whichPNG   = data.stimuli(stimPres(objects));
        data.value_rating.(['trial_',num2str(objects-1)]).context    = 1;
    else       
        data.value_rating.(['trial_',num2str(objects-1)]).whichPNG   = data.stimuli(stimPres(objects)-length(data.stimuli));
        data.value_rating.(['trial_',num2str(objects-1)]).context    = 2;
    end
    
    [xCenter, yCenter] = RectCenter(windowRect);
    baseRect = [0 0 2*cx 2*xy];
    centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
    
    rectColor = [0.5 0.5 0.5];
    % Draw the square to the screen. For information on the command used in
    % this line see Screen FillRect?
    Screen('FillRect', window, rectColor, centeredRect);
    
    drawFixation;
    Screen('Flip', window);

    WaitSecs(1);
    
    data.value_rating.(['trial_',num2str(objects-1)]).time = Screen('Flip', window);
    
    data.value_rating.(['trial_',num2str(objects-1)]).position    = 0.5;%randi(100)/100;
    x = (data.value_rating.(['trial_',num2str(objects-1)]).position * (rightlim-leftlim))+leftlim;
    
    SetMouse(x, windowRect(4)*scalaPosition, window);  
%         
    [x, y, buttons] = GetMouse(window);
    HideCursor;  
    
    rectColor = data.context(data.value_rating.(['trial_',num2str(objects-1)]).context ,:);        
    while sum(buttons) == 0
        
        % Draw the two options to the screen.
        tex = Screen('MakeTexture', window, theImage{stimPres(objects)});
        coin_tex = Screen('MakeTexture', window, coin{data.value_rating.(['trial_',num2str(objects-1)]).context});

        % Make sure the images are appropriately scaled. They should always
        % have a h1ight of 200
        dim = size(theImage{stimPres(objects)});     scaling = dim(1)/400;

        Screen('FillRect', window, rectColor, centeredRect);
        Screen('DrawTexture', window, tex, [], CenterRectOnPoint([0 0 dim(2)/scaling dim(1)/scaling], xCenter, yCenter));

       % endPoints = {sprintf('Keine'), sprintf('Viele')};

        Screen('TextSize', window, 50);
        % Drawing the question as text
        DrawFormattedText(window, 'Wie viele Punkte bekommen Sie für dieses Monster bei dieser Hintergrundfarbe?', 'center', windowRect(4)*0.2, white);

        midTick    = [center(1) windowRect(4)*scalaPosition - lineLength center(1) windowRect(4)*scalaPosition  + lineLength];
        leftTick   = [windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition - lineLength windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition  + lineLength];
        rightTick  = [windowRect(3)*scalaLength windowRect(4)*scalaPosition - lineLength windowRect(3)*scalaLength windowRect(4)*scalaPosition  + lineLength];
        horzLine   = [windowRect(3)*scalaLength windowRect(4)*scalaPosition windowRect(3)*(1-scalaLength) windowRect(4)*scalaPosition];
%         textBounds = [Screen('TextBounds', window, endPoints{1}); Screen('TextBounds', window, endPoints{2})];

        % Drawing the scala
        Screen('DrawLine', window, [1 1 1], midTick(1), midTick(2), midTick(3), midTick(4), width);         % Mid tick
        Screen('DrawLine', window, [1 1 1], leftTick(1), leftTick(2), leftTick(3), leftTick(4), width);     % Left tick
        Screen('DrawLine', window, [1 1 1], rightTick(1), rightTick(2), rightTick(3), rightTick(4), width); % Right tick
        Screen('DrawLine', window, [1 1 1], horzLine(1), horzLine(2), horzLine(3), horzLine(4), width);     % Horizontal line

        % Drawing the end points of the scala as text
        Screen('TextSize', window, 40);
       % DrawFormattedText(window, endPoints{1}, leftTick(1, 1)-50,  windowRect(4)*scalaPosition+40, [],[],[],[],[],[],[]); % Left point
       % DrawFormattedText(window, endPoints{2}, rightTick(1, 1) - textBounds(2, 3)/2+50,  windowRect(4)*scalaPosition+40, [],[],[],[],[],[],[]); % Right point
        
        % Get the current position of the mouse
        [x, y, buttons] = GetMouse(window);
        
        % We clamp the values at the maximum values of the screen in X and Y
        % incase people have two monitors connected. This is all we want to
        % show for this basic demo.
        x = max(min(x, windowRect(3)*scalaLength), round(windowRect(3)*(1-scalaLength)));
        
        pos = (x - leftlim) / (rightlim - leftlim);
        DrawFormattedText(window, num2str(round(pos*100)), x-20, windowRect(4)*scalaPosition-40, white);

                
%         Screen('DrawDots', window, [x windowRect(4)*scalaPosition], dotSizePix, dotColor, [], 2);
        Screen('DrawTexture', window, coin_tex, [], CenterRectOnPoint([0 0 30 30], x, windowRect(4)*scalaPosition));
        
        Screen('Flip', window)
        data.value_rating.(['trial_',num2str(objects-1)]).position(end+1) = pos;
    
    end
    
        
    % Flip to the screen
    data.value_rating.(['trial_',num2str(objects-1)]).time(end+1) = Screen('Flip', window);
        
    data.value_rating.(['trial_',num2str(objects-1)]).RT       = data.value_rating.(['trial_',num2str(objects-1)]).time(end) - data.value_rating.(['trial_',num2str(objects-1)]).time(1);
    
    save ([options.dataPath,'/data_',data.subject,'_',num2str(session),'_',initials,'_postscan.mat'],'data');
end