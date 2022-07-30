function string=dateAndTimeStringNoSec_nk

% returns a string specifying data and time in a format that can be used as
% part of filenames.

c=clock;

year=c(1);
month=c(2);
day=c(3);
hour=c(4);
minute=c(5);
second=c(6);

string=[num2str(year),'-',num2str(month),'-',num2str(day),'_',num2str(hour),'h',num2str(minute),'m'];
