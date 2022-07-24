function [itemPositions,distanceMat_ltv]=letSubjectArrangeItems_circularArena(varargin)
% USAGE
%       [itemPositions,distanceMat_ltv]=letSubjectArrangeItems_circularArena(imageData[,instructionString,options])
%
% FUNCTION
%       This function allows the user to arrange a number of items in a
%       circular "arena" by dragging and dropping with the mouse. Sets
%       of items can be selected by right-clicking single items or
%       right-dragging to draw a selection box. The items are initially
%       placed outside the arena in a "seating" area. The imageData
%       structure array contains the images that represent the items.
%
% ARGUMENTS
% imageData 
%       Structure array with as many entries as there are items to be
%       arranged. The only required field is "image", which must contain
%       the image arrays (to be processed by matlab's image function).
%       Optionally a field "alpha" can be added to control the alpha
%       channel, i.e. to define transparent regions (for details see help
%       on matlab's image function).
%
% [instructionString]
%       Optional string containing the instruction for the subject, such as
%       "Please arrange the items according to their visual similarity?".


%% define GUI and handle the case of a GUI callback 
if nargin
    % initialization code
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       'letSubjectArrangeItems_circularArena', ...
        'gui_Singleton',  gui_Singleton, ...
        'gui_OpeningFcn', @letSubjectArrangeItems_circularArena_OpeningFcn, ...
        'gui_OutputFcn',  @letSubjectArrangeItems_circularArena_OutputFcn, ...
        'gui_LayoutFcn',  [] , ...
        'gui_Callback',   []);

    if ischar(varargin{1}) % it's a callback from the GUI
        gui_State.gui_Callback = str2func(varargin{1});
        gui_mainfcn(gui_State, varargin{:});
        return;
    end
end

% control passes here only on initial user call

%% define options
if nargin>2
    options=varargin{3};
else
    options.axisUnits='normalized';
end

options=setIfUnset(options,'axisUnits','normalized');
options=setIfUnset(options,'requireTrialConfirmation',false);
ud.options=options;



%% open GUI and store information in GUI userdata (ud)
% open GUI
[hf,ha,hpb1,hpb2] = gui_mainfcn(gui_State, varargin{:}); % opens the GUI window and returns GUI info

% position the window
lbwh = get(0,'ScreenSize');
set(hf,'Units','pixels'); 
windowReductionFactor=.8;
lbwh_fig=[lbwh(3:4)*((1-windowReductionFactor)/2) lbwh(3:4)*windowReductionFactor];
set(hf,'Position',lbwh_fig);
drawnow;

% handle arguments
ud.imageData=varargin{1};
if numel(ud.imageData)<2
    error('letSubjectArrangeItems_circularArena: pass at least two images to represent the items to be arranged.');
end

if nargin>1
    ud.instructionString=varargin{2};
else
    ud.instructionString='[subject instruction string here]'; 
end

% arena size
ud.nItems=numel(ud.imageData);
[ud.imHeight,ud.imWidth,ignore]=size(ud.imageData(1).image); % all images assumed to be of the same size
ud.squareObjWidth=max(ud.imHeight,ud.imWidth); % the axes unit is pixel (in image bmp, not on the screen)
nItemsThatWouldFillTheArena=ud.nItems*3;

if strcmp(ud.options.axisUnits,'Pixels')  % ud.options.axisUnits=='Pixels'
    % CONSTANT FIGURE AND IMAGE SIZE
    % the items have constant size: they appear at the screen resolution
    % (one screen pixel equals one imageData pixel).
    % in this mode the window is not resizable.
    set(hf,'Resize','off');

    ud.arenaMargin=ud.squareObjWidth*0.15;
    ud.squareAxisWidth=min(lbwh_fig(3:4));
    ud.arenaRadius=ud.squareAxisWidth/2-ud.arenaMargin-ud.squareObjWidth;
    
else
    % FIGURE RESIZABLE, ARENA AND IMAGES RESIZED IN PROPORTION
    set(hf,'Resize','on');
    
    % r^2*pi=n*ow^2 => r=sqrt(n*ow^2/pi)
    %(r: arena radius, n: number of items filling the arena, ow: item width)
    ud.arenaRadius=sqrt(nItemsThatWouldFillTheArena*ud.squareObjWidth^2/pi);

    ud.arenaMargin=ud.squareObjWidth*0.15;
    ud.squareAxisWidth=2*(ud.arenaRadius+ud.arenaMargin+ud.squareObjWidth);
end

ud.ha=ha; % handle of axis
ud.hpb1=hpb1; % handle of push button 1
ud.hpb2=hpb2; % handle of push button 2

set(hf,'UserData',ud);


%% adjust GUI properties
%opengl software; 
set(hpb1,'TooltipString','Press here to clear the arena and start over.');
set(hpb2,'TooltipString','Press here to indicate that you are finished.');
set(hpb2,'Enable','off');
%set(hf,'Renderer','Painters'); 
%set(hf,'Renderer','ZBuffer'); 
set(hf,'Renderer','OpenGL'); % OpenGL appears to be the default anyway, so this line is redundant

% Note: 
% There's a bug in Matlab (version 2009a) that causes text to flip when you 
% use the 'OpenGL' rendering option. 
% See <http://www.mathworks.com/matlabcentral/answers/210> for more 
% information. The text flipping issue can be resolved by typing 
% <opengl software>, but this will result in other display problems.
% The other rendering options don't give these problems, but they do not 
% use the alpha channel.

set(hf,'KeyPressFcn','keypress_Callback');


%% create "arrangement arena"
initializeArena(hf);
ud=get(hf,'UserData');


%% wait for the subject to arrange the items 
tic % start stopwatch to time the subject
button='No, I''ll adjust the arrangement.';
while ~strcmp(button,'Yes, I am done.')
    ud.donePressed=false; set(hf,'UserData',ud);
    while ~ud.donePressed
        if allInsideArena(hf)
            set(hpb2,'Enable','on'); % enable "done" button
        else
            set(hpb2,'Enable','off'); % disable "done" button
        end
        pause(0.1); % wait 0.1 s
        ud=get(hf,'UserData');
    end
    
    % subject has pressed done
    if ud.options.requireTrialConfirmation
        button = questdlg('Are you sure you are done arranging the items?','','Yes, I am done.','No, I''ll adjust the arrangement.','No, I''ll adjust the arrangement.');
    else
        button='Yes, I am done.';
    end
end
trialTimeDuration=toc;
% the subject has indicated again that the arrangement is final.


%% return the final arrangement
itemPositions=nan(ud.nItems,2);
for itemI=1:ud.nItems
    xdata=get(ud.h_image(itemI),'XData');
    ydata=get(ud.h_image(itemI),'YData');
    x=xdata(1)+ud.imWidth/2-ud.ctrXY;
    y=ydata(1)+ud.imHeight/2-ud.ctrXY;
    itemPositions(ud.seatingOrder(itemI),:)=[x,y];
end

itemPositions=itemPositions./(ud.arenaRadius*2);
% scale such that the arena's diameter corresponds to 1

distanceMat_ltv=pdist(itemPositions,'euclidean');


%% save the results to ensure that it is never lost
% trialIDstring=datestr(clock,30);
% save(['itemPositions_circularArena_',trialIDstring,'.txt'],'itemPositions','-ascii');
% save(['distanceMat_ltv_circularArena_',trialIDstring,'.txt'],'distanceMat_ltv','-ascii');
% save(['itemPositions_distanceMat_ltv_circularArena_',trialIDstring,'.mat'],'itemPositions','distanceMat_ltv','trialTimeDuration');
savefig(options.figname)
close(hf);

% function returns control



%% --------------------------------------------------------------------------
function initializeArena(hf)
moveImages('reset');

ud=get(hf,'UserData');

cla(ud.ha);
axis(ud.ha,'equal','off');
set(ud.ha,'Units',ud.options.axisUnits);


title(ud.ha,ud.instructionString,'FontUnits','normalized','FontSize',.03,'FontWeight','bold');
set(hf,'Color',[.9 .9 .9]);
axis(ud.ha,[0 ud.squareAxisWidth 0 ud.squareAxisWidth]);
set(ud.ha,'YDir','reverse'); % y axis points down (for image display)

% draw circular arena
ud.h_arena=rectangle('Position',[ud.squareObjWidth+ud.arenaMargin ud.squareObjWidth+ud.arenaMargin 2*ud.arenaRadius 2*ud.arenaRadius],'Curvature',[1 1],'EdgeColor','none','FaceColor',[1 1 1]);

% group selection of items: add callbacks to axes item
set(ud.h_arena,'ButtonDownFcn','moveImages(''buttonDown'',''ellipse'')');

% arrange images in random sequence in peripheral circle
ud.seatingOrder=randperm(ud.nItems);
initArrangementRad=ud.arenaRadius+ud.arenaMargin+ud.squareObjWidth/2;
ud.ctrXY=ud.squareAxisWidth/2;
angles_rad = 2*pi*(rand + (0:ud.nItems-1)/ud.nItems);
x = cos(angles_rad)*initArrangementRad;
y = sin(angles_rad)*initArrangementRad;
ud.h_image=nan(ud.nItems,1);

for itemI=1:ud.nItems
    ud.h_image(itemI)=image('XData',ud.ctrXY+x(itemI)-ud.imWidth/2,...
        'YData',ud.ctrXY+y(itemI)-ud.imHeight/2,...
        'CData',ud.imageData(ud.seatingOrder(itemI)).image,...
        'ButtonDownFcn','moveImages(''buttonDown'',''ellipse'')');

    % use imageData field 'alpha' to define transparency (if alpha exists and is not empty)
    if isfield(ud.imageData(ud.seatingOrder(itemI)),'alpha') && ~isempty(ud.imageData(ud.seatingOrder(itemI)).alpha)
        set(ud.h_image(itemI),'AlphaData',ud.imageData(ud.seatingOrder(itemI)).alpha);
    end
end

set(hf,'UserData',ud);


%% --------------------------------------------------------------------------
function answer=allInsideArena(hf)
ud=get(hf,'UserData');

insideArena=false(ud.nItems,1);
for itemI=1:ud.nItems
    xdata=get(ud.h_image(itemI),'XData');
    ydata=get(ud.h_image(itemI),'YData');
    x=xdata(1)+ud.imWidth/2-ud.ctrXY;
    y=ydata(1)+ud.imHeight/2-ud.ctrXY;
    insideArena(itemI)=sqrt(x^2+y^2)<ud.arenaRadius;
end
answer=all(insideArena);



%% --------------------------------------------------------------------------
% --- Executes just before letSubjectArrangeItems_circularArena is made visible.
function letSubjectArrangeItems_circularArena_OpeningFcn(hItem, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hItem    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to letSubjectArrangeItems_circularArena (see VARARGIN)

% Choose default command line output for letSubjectArrangeItems_circularArena
handles.output = hItem;

% Update handles structure
guidata(hItem, handles);

% UIWAIT makes letSubjectArrangeItems_circularArena wait for user response (see UIRESUME)
% uiwait(handles.figure1);


%% --------------------------------------------------------------------------
% --- Outputs from this function are returned to the command line.
function varargout = letSubjectArrangeItems_circularArena_OutputFcn(hItem, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hItem    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
varargout{2} = handles.axes1;
varargout{3} = handles.pushbutton1;
varargout{4} = handles.pushbutton2;


%% --------------------------------------------------------------------------
% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hItem, eventdata, handles)
% hItem    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

initializeArena(handles.figure1);


%% --------------------------------------------------------------------------
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hItem, eventdata, handles)
% hItem    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ud=get(handles.figure1,'UserData');
ud.donePressed=true;
set(handles.figure1,'UserData',ud);


%% --------------------------------------------------------------------------
% --- Executes on any key press ocurring while GUI has the focus
function keypress_Callback
% disp('keypress_Callback');

scaleFactor=1.02;

moveImages('reset');

hf=gcbf;
ud=get(hf,'UserData');

lastKey=get(hf,'CurrentCharacter');

if any(lastKey=='aAzZ')
    % determine item positions (relative to center of arena)
    itemPositions=nan(ud.nItems,2);
    for itemI=1:ud.nItems
        xdata=get(ud.h_image(itemI),'XData');
        ydata=get(ud.h_image(itemI),'YData');
        x=xdata(1)+ud.imWidth/2-ud.ctrXY;
        y=ydata(1)+ud.imHeight/2-ud.ctrXY;
        itemPositions(ud.seatingOrder(itemI),:)=[x,y];
    end
    
    % apply the scale factor to the arrangement
    if any(lastKey=='aA')
        itemPositions=itemPositions*scaleFactor;
    elseif any(lastKey=='zZ')
        itemPositions=itemPositions/scaleFactor;
    end
    
    xy_new=(itemPositions+ud.ctrXY)-repmat([ud.imWidth/2,ud.imHeight/2],[ud.nItems 1]);
    
    xlim=get(ud.ha,'XLim');
    ylim=get(ud.ha,'YLim');
    
    if all(xlim(1)<xy_new(:,1)) && all(xy_new(:,1)<xlim(2)) &&...
       all(ylim(1)<xy_new(:,2)) && all(xy_new(:,2)<ylim(2)),
        % all items are still entirely within the axes limits: move all (otherwise: move none.)
        for itemI=1:ud.nItems
            set(ud.h_image(itemI),'XData',xy_new(ud.seatingOrder(itemI),1));
            set(ud.h_image(itemI),'YData',xy_new(ud.seatingOrder(itemI),2));
        end
    end % item within axes limits
end % any(lastKey=='aAzZ')

set(hf,'UserData',ud);

