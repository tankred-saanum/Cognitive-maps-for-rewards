%% Many maps experiment
% 
% saved as data_#.mat in ../datafiles
%
%__________________________________________________________________________
% Copyright (C) 2018 Mona M Garvert, MPI CBS Leipzig

% Clear the workspace and the
% screen
close all;
clear Screen
clearvars;

%% Define variables
% ====================================================================a=====
defineOptions;

%% Start Psychtoolbox and setup experiment
% =========================================================================
setup
[xCenter, yCenter] = RectCenter(windowRect);
    
%% Welcome screen
% =========================================================================
KbName('UnifyKeyNames')
 
if ~options.test
    HideCursor;
end

% Set the text
textString = 'Herzlich Willkommen!';
Screen('TextSize', window, 30);
    
% Draw the word
Screen('DrawText',window, textString, xCenter*2/3, yCenter, white);
Screen('Flip', window);     % Flip to the screen

% The experimenter starts the script by pressing 'A' then 'C'
[~,key] = KbPressWait;

while find(key)~=KbName('A')
    disp(find(key))
    [~,key] = KbPressWait;
end
while find(key)~=KbName('C')
    [~,key] = KbPressWait;
end
Screen('Flip', window);     % Flip to the screen

%% Button test
% =========================================================================

Screen('TextSize', window, 15);
% textString = sprintf('Wir testen jetzt die Tasten.\n\n\n\nBitte drücken Sie eine der beiden Tasten um zu starten.');
textString1 = sprintf('Wir testen jetzt die Tasten.');
textString2 = sprintf('Bitte drücken Sie eine der beiden Tasten um zu starten.');
% DrawFormattedText(window, textString, 'center', 'center', white);
Screen('DrawText',window, textString1, xCenter*1/5, yCenter, white);
Screen('DrawText',window, textString2, xCenter*1/5, yCenter+yCenter/10, white);

Screen('Flip', window);

% Flip to the screen

if ~buttonbox
    leftKey     = KbName('LeftArrow');
    centreKey   = KbName('DownArrow');
    rightKey    = KbName('RightArrow');
    KbPressWait
elseif options.cbs
    leftKey     = 252;
    rightKey    = 248;
    key = [];
    while (isempty(key))
        IOPort('Purge',button_port)
        [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
    end
else
    leftKey     = 1;
    rightKey    = 2;
    [tKeyPress,key] = IOWaitButton(IO,inf,ioObj,address);
end
                           
textString = 'Bitte drücken Sie die linke Taste.';
Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
Screen('Flip', window);     % Flip screen

if ~buttonbox
    [~,key] = KbPressWait;
    checkEscape
    while find(key)~=leftKey
        textString = sprintf('Das war die falsche Taste. Bitte drücken Sie die linke Taste.');
%         DrawFormattedText(window, textString, 'center', 'center', white);
        Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
        Screen('Flip', window);     % Flip screen
        [~,key] = KbPressWait;
        checkEscape
    end
else
    key = [];
    while (isempty(key))
        IOPort('Purge',button_port)
        [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
    end
    
    checkEscape
    while key~=leftKey
        disp(key)
        textString = sprintf('Das war die falsche Taste. Bitte drücken Sie die linke Taste.');
        Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
        Screen('Flip', window);     % Flip screen
        key = [];
        while (isempty(key))
            IOPort('Purge',button_port)
            [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
        end
        checkEscape
    end
end

textString = 'Bitte drücken Sie die rechte Taste.';
Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
Screen('Flip', window);     % Flip screen

if ~buttonbox
    [~,key] = KbPressWait;
    checkEscape
    while find(key)~=rightKey
        textString = sprintf('Das war die falsche Taste. Bitte drücken Sie die rechte Taste.');
        Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
        Screen('Flip', window);     % Flip screen
        [~,key] = KbPressWait;
        checkEscape
    end
else
    key = [];
    while (isempty(key))
        IOPort('Purge',button_port)
        [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
    end
    
    checkEscape
    while key~=rightKey
        disp(key)
        textString = sprintf('Das war die falsche Taste. Bitte drücken Sie die rechte Taste.');
        Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
        Screen('Flip', window);     % Flip screen
        key = [];
        while (isempty(key))
            IOPort('Purge',button_port)
            [key,tKeyPress] = IOPort('Read', button_port, 1, 1);
        end
        checkEscape
    end
end


textString = 'Vielen Dank. Bitte warten Sie.';
Screen('DrawText',window, textString, xCenter*1/5, yCenter, white);
Screen('Flip', window);     % Flip screen
[~,key] = KbPressWait;


%% Training phase
% =========================================================================
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
disp('START SPIKE')
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')        

data.choice.payout = 0;

if session == 3 && ~strcmp(restart,'j')
    % Run the decision making block, only in session 3
    run_choice_block

    Screen('TextSize', window, 20);
    textString1 = '       Ende des Blocks';
    textString2 = 'Verdienst in diesem Block:';
    textString3 = sprintf('            %.2f Euro',data.choice.payout);
    Screen('DrawText',window, textString1, xCenter*4/5, yCenter, white);
    Screen('DrawText',window, textString2, xCenter*4/5, yCenter+yCenter/10, white);
    Screen('DrawText',window, textString3, xCenter*4/5, yCenter+2*yCenter/10, white);
    Screen('Flip', window);     % Flip screen
    [~,key] = KbPressWait;
end
  
for bl = startblock:options.scanblocks

    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('START SPIKE')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
    data.scan{bl}.RSA.objDiff.payout = 0;
    
    % Run the RSA block
    run_RSA_block
    
    fprintf('Payout this block: %.2f\n',data.scan{bl}.RSA.objDiff.payout);
    
    
    data.scan{bl}.endTime  =  GetSecs;

    Screen('TextSize', window, 20);
    textString1 = '       Ende des Blocks';
    textString2 = 'Verdienst in diesem Block:';
    textString3 = sprintf('            %.2f Euro',data.scan{bl}.RSA.objDiff.payout);
    Screen('DrawText',window, textString1, xCenter*4/5, yCenter, white);
    Screen('DrawText',window, textString2, xCenter*4/5, yCenter+yCenter/10, white);
    Screen('DrawText',window, textString3, xCenter*4/5, yCenter+2*yCenter/10, white);
    Screen('Flip', window); 
    
    if options.scan
        [~,key] = KbPressWait;
        while find(key)~=KbName('A')
            disp(find(key))
            [~,key] = KbPressWait;
        end
        while find(key)~=KbName('C')
            [~,key] = KbPressWait;
        end
    else
        [~,key] = KbPressWait;
    end
end

textString1 = 'ENDE';
Screen('DrawText',window, textString1, xCenter, yCenter, white);
Screen('Flip', window);     % Flip screen

%data.comments = input('Comments? ','s');
saveData

% The experimenter starts the script by pressing 'A' then 'C'
[~,key] = KbPressWait;
while find(key)~=KbName('A')
    disp(find(key))
    [~,key] = KbPressWait;
end
while find(key)~=KbName('C')
    [~,key] = KbPressWait;
end
Screen('Flip', window);     % Flip to the screen

sca;

total_payout = data.choice.payout;
for bl = 1:options.scanblocks
   total_payout = total_payout + data.scan{bl}.RSA.objDiff.payout;
end 
fprintf('Total payout: %.2f',total_payout);

