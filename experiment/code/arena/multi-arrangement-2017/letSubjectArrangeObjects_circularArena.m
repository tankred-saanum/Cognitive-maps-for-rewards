% workaround: For some reason the Callback of the 'Done' PushButton in the GUI is still registered
% to 'letSubjectArrangeObjects_circularArena'. For another reason this
% cannot be changed in the fig-file. We thus use a little, dirty workaround
% here and just redirect 'letSubjectArrangeObjects_circularArena(varargin)' to 'letSubjectArrangeItems_circularArena(varargin)'
function letSubjectArrangeObjects_circularArena(varargin)    
    letSubjectArrangeItems_circularArena(varargin{:});