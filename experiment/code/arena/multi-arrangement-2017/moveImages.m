function moveImages(action,groupSelectionShape)
% USAGE
%       moveImages(action[,groupSelectionShape])
%
% FUNCTION
%       serves to allow the user to mouse-drag groups of images within axes.
%       clicking with the right mouse button toggles selection of images to
%       be moved together. dragging with the right mouse button draws a
%       rectangle or ellipse for selecting groups of images.
%
%       moveImages needs to be defined as a button-down callback function for each
%       image to be moved by the user, e.g. as follows:
%       set(h_image,'ButtonDownFcn','moveImages(''buttonDown'')');.
%
%       for group selection to work, the background object as well needs to
%       have a button-down callback of moveImages defined, e.g. as follows: 
%       set(h_backgroundAxesOrRectangle,'ButtonDownFcn','moveImages(''buttonDown'')');
%
%       (nikolaus kriegeskorte, jan 2008)
%
% ARGUMENTS
% action
%       the name of of the subfunction of the callback function moveImages
%       to be invoked. callbacks need to be defined only for
%       action='buttonDown'. the other subfunctions (actions) are defined
%       automatically by that action.
%
% [groupSelectionShape]
%       this optional argument can be either 'rectangle' (default) or
%       'ellipse'. it defines the shape that is used to select groups of
%       images by means of right-button dragging.


%% preparations
if ~exist('groupSelectionShape','var')
    groupSelectionShape='rectangle'; % alternative: 'ellipse'
end


%% invoke subfunction passed as argument
feval(action,groupSelectionShape);
return



%% start new selection shape (when mouse button is pressed)
function buttonDown(groupSelectionShape)
% disp('button down'); %get(gco)
hf = gcbf; % handle of clicked figure
ho=gcbo; % handle of clicked object
ud = get(hf,'UserData'); % state of operation stored in the figure's user data

set(hf,'WindowButtonMotionFcn',['moveImages(''mouseMove'',''',groupSelectionShape,''')']);
set(hf,'WindowButtonUpFcn',['moveImages(''buttonUp'',''',groupSelectionShape,''')']);

cp=get(ud.ha,'CurrentPoint'); cp=cp(1,1:2);

if ~isfield(ud,'hs_selectedImages')
    ud.hs_selectedImages=[];
end

button = get(hf,'SelectionType');
switch button
    case 'normal'
        if strcmp(get(ho,'Type'),'image')
            % (1) subject left-clicked an image

            if strcmp(get(ho,'Selected'),'off');
                % subject left-clicked an unselected image: clear selection and select only that image

                % clear selection
                for selectedImageI=1:numel(ud.hs_selectedImages)
                    set(ud.hs_selectedImages(selectedImageI),'Selected','off')
                end
                % select current image only
                ud.hs_selectedImages=ho;
                set(ho,'Selected','on');
            end
            
            % get ready to move all selected images
            ud.mouseMode='movingImages';
        end
    
    case 'alt'
        if strcmp(get(ho,'Type'),'image');
            % (2a) subject right-clicked an image: toggle image selection

            if strcmp(get(ho,'Selected'),'on')
                % image is selected: unselect it
                set(ho,'Selected','off');
                ud.hs_selectedImages(ud.hs_selectedImages==ho)=[];
            else
                % image is unselected: select it
                set(ho,'Selected','on');
                ud.hs_selectedImages=[ud.hs_selectedImages,ho];
            end            
        else
            % (2b) subject right-clicked elsewhere: start new selection shape
            ud.mouseMode='drawingSelectionShape';
            ud.hs_imagesInCurrentShape=[];
            ud.hs_previouslySelectedImages=ud.hs_selectedImages;
            
            ud.shapeBoundingBoxCorner1=cp;

            % delete previous selection shape (if it exists)
            if isfield(ud,'selectionShapeHandle')
                delete(ud.selectionShapeHandle);
            end

            % create new selection shape
            if strcmp(groupSelectionShape,'ellipse')
                curvature=[1 1];
            else
                curvature=[0 0];
            end
            ud.selectionShapeHandle=rectangle('Position',[cp(1),cp(2),0.001,0.001],'Curvature',curvature,...
                'FaceColor','none','LineWidth',3,'EdgeColor',[.7 .7 .7],'LineStyle','--');
            % disp('created new selection shape...'); ud.selectionShapeHandle
        end
end %switch
ud.previousPoint=cp;

set(hf,'UserData',ud);
% disp('~button down'); ud


%% drag to select or move images
function mouseMove(groupSelectionShape)
% disp('mouse move'); %get(gco)
hf = gcbf;
ud = get(hf,'UserData'); % state of operation stored in the figure's user data

if ~isfield(ud,'hs_selectedImages')
    ud.hs_selectedImages=[];
end

cp=get(ud.ha,'CurrentPoint'); cp=cp(1,1:2);
hs_objectsInAxes=get(ud.ha,'Children');

if isfield(ud,'mouseMode')
    % ud.mouseMode
    switch ud.mouseMode
        case 'drawingSelectionShape'
            if ~isfield(ud,'selectionShapeHandle') || isempty(ud.selectionShapeHandle) || ~ishandle(ud.selectionShapeHandle)
                reset(groupSelectionShape);
            else
                % adjust the selection shape's size
                ud.shapeBoundingBoxCorner1;
                ud.shapeBoundingBoxCorner2=cp;
                xy=min([ud.shapeBoundingBoxCorner1;ud.shapeBoundingBoxCorner2],[],1);
                wh=abs(ud.shapeBoundingBoxCorner1-ud.shapeBoundingBoxCorner2);
                if all(wh>0)
                    set(ud.selectionShapeHandle,'Position',[xy(1) xy(2) wh(1) wh(2)]);
                end

                % select all images within the shape
                shapeCenterXY=mean([ud.shapeBoundingBoxCorner1;ud.shapeBoundingBoxCorner2],1);

                ud.hs_imagesInCurrentShape=[];
                for objectI=1:numel(hs_objectsInAxes)
                    h_cObj=hs_objectsInAxes(objectI); % handle of current object
                    if strcmp(get(h_cObj,'Type'),'image')
                        % this object is an image

                        xdata=get(h_cObj,'XData');
                        ydata=get(h_cObj,'YData');
                        cdata=get(h_cObj,'CData');
                        [imHeight,imWidth,ignore]=size(cdata); % images may have different sizes

                        x=xdata(1)+imWidth/2-shapeCenterXY(1);
                        y=ydata(1)+imHeight/2-shapeCenterXY(2);
                        radXY=wh/2;

                        if strcmp(groupSelectionShape,'ellipse')
                            imageInShape=(x/radXY(1))^2+(y/radXY(2))^2<1;
                        else
                            imageInShape= -radXY(1)<x && x<radXY(1) &&...
                                -radXY(2)<y && y<radXY(2);
                        end

                        if imageInShape
                            % this image is within the current shape: select it
                            ud.hs_imagesInCurrentShape=[ud.hs_imagesInCurrentShape,h_cObj];
                            if strcmp(get(h_cObj,'Selected'),'off')
                                set(h_cObj,'Selected','on');
                                ud.hs_selectedImages=[ud.hs_selectedImages,h_cObj];
                            end
                        else
                            % this image is without the current shape...
                            if ~any(ud.hs_previouslySelectedImages==h_cObj)
                                % ...and was not previously selected: unselect it
                                set(h_cObj,'Selected','off');
                                ud.hs_selectedImages(ud.hs_selectedImages==h_cObj)=[];
                                ud.hs_imagesInCurrentShape(ud.hs_imagesInCurrentShape==h_cObj)=[];
                            end
                        end
                    end
                end % objectI loop
            end % ~(~isfield(ud,'selectionShapeHandle') || isempty(ud.selectionShapeHandle))

        case 'movingImages'
            % move all selected images
            motionVector=cp-ud.previousPoint;

            for objectI=1:numel(hs_objectsInAxes)
                h_cObj=hs_objectsInAxes(objectI); % handle of current object
                if strcmp(get(h_cObj,'Type'),'image') && strcmp(get(h_cObj,'Selected'),'on')
                    % this object is a selected image: move it
                    xdata=get(h_cObj,'XData');
                    ydata=get(h_cObj,'YData');

                    xdata=xdata+motionVector(1);
                    ydata=ydata+motionVector(2);
                    
                    xlim=get(ud.ha,'XLim');
                    ylim=get(ud.ha,'YLim');

                    cdata=get(h_cObj,'CData');
                    [imHeight,imWidth,ignore]=size(cdata); % images may have different sizes
                    
                    xdata=max(xdata,xlim(1));
                    xdata=min(xdata,xlim(2)-imWidth);
                    ydata=max(ydata,ylim(1));
                    ydata=min(ydata,ylim(2)-imHeight);
                        
                    % object will be completely within axes in the new position
                    set(h_cObj,'XData',xdata);
                    set(h_cObj,'YData',ydata);
                end
            end % objectI loop
    end % switch ud.mouseMode
end
ud.previousPoint=cp;

set(hf,'UserData',ud);
% disp('~mouse move'); ud


%% finish selection shape (when mouse button is released)
function buttonUp(groupSelectionShape)
% disp('button up'); %get(gco)
hf=gcbf;
ud = get(hf,'UserData');

% clear selection if mouse mode was "drawing selection shape" and shape was empty
if isfield(ud,'mouseMode') && strcmp(ud.mouseMode,'drawingSelectionShape') && isfield(ud,'hs_imagesInCurrentShape') && numel(ud.hs_imagesInCurrentShape)==0
    % the selection shape did not contain any images:
    % unselect all objects
    if isfield(ud,'hs_selectedImages')
        for selectedImageI=1:numel(ud.hs_selectedImages)
            set(ud.hs_selectedImages(selectedImageI),'Selected','off');
        end
    end
    ud.hs_selectedImages=[];
end

% clear selection shape
if isfield(ud,'selectionShapeHandle') && ~isempty(ud.selectionShapeHandle) && ishandle(ud.selectionShapeHandle)
    delete(ud.selectionShapeHandle);
end
ud.selectionShapeHandle=[];
ud.hs_imagesInCurrentShape=[];

% hover unencumbered
ud.mouseMode='hoveringUnencumbered';

set(hf,'UserData',ud);
% disp('~button up');



%% reset: clear selection and hover unencumbered
function reset(groupSelectionShape)
hf=gcbf;
ud = get(hf,'UserData');

% clear selection shape
if isfield(ud,'selectionShapeHandle') && ~isempty(ud.selectionShapeHandle) && ishandle(ud.selectionShapeHandle)
    delete(ud.selectionShapeHandle);
end
ud.selectionShapeHandle=[];
ud.hs_imagesInCurrentShape=[];

% unselect all objects
if isfield(ud,'hs_selectedImages')
    for selectedImageI=1:numel(ud.hs_selectedImages)
        set(ud.hs_selectedImages(selectedImageI),'Selected','off');
    end
end
ud.hs_selectedImages=[];

% hover unencumbered
ud.mouseMode='hoveringUnencumbered';

set(hf,'UserData',ud);
