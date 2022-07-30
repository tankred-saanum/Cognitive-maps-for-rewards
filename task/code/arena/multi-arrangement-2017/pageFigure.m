function h=pageFigure(figI,paperSize,proportionOfScreenArea,horPos0123);
% h=pageFigure([figI,paperSize,proportionOfScreenArea,horPos1234]);

if ~exist('figI'), figI=gcf; end
%if ~exist('paperSize'), paperSize='A4'; end
if ~exist('paperSize')||isempty(paperSize), paperSize='letter'; end
%if ~exist('proportionOfScreenArea')||isempty(proportionOfScreenArea), proportionOfScreenArea=0.23; end
if ~exist('proportionOfScreenArea')||isempty(proportionOfScreenArea), proportionOfScreenArea=.5; end

if ishghandle(figI)
    h=figure(figI(1));
else
    h=figure;
end

if ~exist('horPos0123'), horPos0123=mod(mod(h.Position(4),10),4); end

set(h,'Color','w');
%set(h,'WindowStyle','docked');

if strcmp(paperSize,'A4')
    heightToWidth=sqrt(2)/1;
elseif strcmp(paperSize,'legal')
    heightToWidth=14/8.5;
elseif strcmp(paperSize,'letter')
    heightToWidth=11/8.5;
end

lbwh = get(0,'ScreenSize');
screenArea=lbwh(3)*lbwh(4);

figWidth=sqrt(screenArea*proportionOfScreenArea/heightToWidth);
figHeight=heightToWidth*figWidth;

left=lbwh(3)/2*horPos0123;
% left=(lbwh(3)-figWidth)/2;
bottom=(lbwh(4)-figHeight)/2;

set(h,'Position',[left bottom figWidth figHeight])
% [left, bottom, width, height]

set(h,'PaperPositionMode','auto'); % 'auto' here prevents resizing when the figure is printed.
