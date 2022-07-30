% local FUNCTION: addHeadingAndPrint(heading)
function addHeadingAndPrint(heading,filespec,figI)

if ~exist('figI','var'), figI=gcf; end
pageFigure(figI);

h=axes('Parent',gcf); hold on;
set(h,'Visible','off');
axis([0 1 0 1]);

% add heading(s)
text(1.11,1.08,heading,'HorizontalAlignment','Right','VerticalAlignment','Top','FontSize',14,'FontWeight','bold','Color','k');

% print as postscript file (appending)
print('-dpsc2','-append',filespec);
%print('-dpsc2',[filespec,'_lastFig']);