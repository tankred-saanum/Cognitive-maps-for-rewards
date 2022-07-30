function str=any2str(varargin)

% FUNCTION
%       concatenates strings and numbers into a single string. by default,
%       numbers are converted to strings with 3-digit precision. the
%       precision (number of digits) can be passed as final argument in the
%       form of a cell (to indicate that it is not to be included in the
%       concatenation.
%
% USAGE EXAMPLES
%       disp(any2str('a=',a,'b=',b,'c=',c))
%       disp(any2str('a=',a,'b=',b,'c=',c,{9}))

if iscell(varargin{nargin})
    precision=cell2mat(varargin{nargin});
else
    %precision=3;
    precision='%6.3f';
end

str=[];
for argI=1:nargin
    if isa(varargin{argI},'char')
        str=[str,varargin{argI}];
    elseif isnumeric(varargin{argI})
        precision=min(numel(num2str(floor(varargin{argI}),8))+2,4);
        str=[str,num2str(varargin{argI},precision)];
    end
end