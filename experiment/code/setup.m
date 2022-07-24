%% Start Psychtoolbox
% =========================================================================

%% Settings
bg                  = 0.3;            % Screen background colour
fontcol             = [1 1 1];        % foreground colour (optional)
fontName            = 'Helvetica'; 
fontSize            = 20;
screenMode          = options.screensize;     % 0 for small window, 1 for full screen, 2 for second screen if attached
number_of_buffers   = 9;              % how many offscreen buffers to create- FOR ACCURATE TIMING OF STIM PRESENTATION


%%
% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% Get the screen numbers. This gives us a number for each of the screens
% attached to our computer. For example, when I call this I get the vector
% [0 1]. The first number is the native display for my laptop and the
% second referes to my secondary external monitor. By native display I mean
% the display the is physically part of my laptop. With a non-laptop
% computer look at your screen preferences to see which is the primary
% monitor.
screens = Screen('Screens');

% To draw we select the maximum of these numbers. So in a situation where we
% have two screens attached to our monitor we will draw to the external
% screen. If I were to select the minimum of these numbers then I would be
% displaying on the physical screen of my laptop.
screenNumber = max(screens);

% Define black and white (white will be 1 and black 0). This is because
% in general luminace values are defined between 0 and 1 with 255 steps in
% between. All values in Psychtoolbox are defined between 0 and 1
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);

%% Screen
% Open an on screen window and color it bg. This function returns a
% number that identifies the window we have opened "window" and a vector
% "windowRect".
% "windowRect" is a vector of numbers: the first is the X coordinate
% representing the far left of our screen, the second the Y coordinate
% representing the top of our screen,
% the third the X coordinate representing
% the far right of our screen and finally the Y coordinate representing the
% bottom of our screen.
grey = white / 2;
%PsychDebugWindowConfiguration(0, 0.5)
if options.screensize == 0
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey,[1,1,2400,1802], 32, 2);
else
    [window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);
end
    
% Get the size of the on screen window in pixels, these are the last two
% numbers in "windowRect" and "rect"
[scr.Xpixels, scr.Ypixels] = Screen('WindowSize', window);

% Get the centre coordinate of the window in pixels.
% xCenter = screenXpixels / 2
% yCenter = screenYpixels / 2
[scr.xCenter, scr.yCenter] = RectCenter(windowRect);

% Query the inter-frame-interval. This refers to the minimum possible time
% between drawing to the screen
scr.ifi = Screen('GetFlipInterval', window);

% We can also determine the refresh rate of our screen. The
% relationship between the two is: ifi = 1 / hertz
scr.hertz = FrameRate(window);

% Here we get the pixel size. This is not the physical size of the pixels
% but the color depth of the pixel in bits
scr.pixelSize = Screen('PixelSize', window);

% Queries the display size in mm as reported by the operating system. Note
% that there are some complexities here. See Screen DisplaySize? for
% information. So always measure your screen size directly.
[scr.width, scr.height] = Screen('DisplaySize', screenNumber);

% Get the maximum coded luminance level (this should be 1)
scr.maxLum = Screen('ColorRange', window);

% Retreive the maximum priority number and set max priority
topPriorityLevel = MaxPriority(window);
Priority(topPriorityLevel);

% Priority(0); % at the end

% Setup the text type for the window
Screen('TextFont', window, fontName);
Screen('TextSize', window, fontSize);

% config_keyboard; 
% config_display(screenMode, screenRes, bg ,fontcol, fontName, fontSize, number_of_buffers,0);   % open graphics window
% config_serial(data.scanTech.scanPort)

data.when_start     = datestr(now,0);
% start_cogent;


% Connect to ports
if options.cbs
    IOPort('CloseAll')
    button_port = IOPort('OpenSerialPort',data.scanTech.buttonPort,'ReceiveTimeout=1');
    trigger_port = IOPort('OpenSerialPort',data.scanTech.triggerPort);
    buttonbox = 1;
elseif options.scan
    open_IOport;
    IOport_logic; %defines IO.trig, IO.def, IO.buttons (4 rows)
    buttonbox     = 1;
else
    buttonbox     = 0;
end

if options.test
    buttonbox = 0;
end
