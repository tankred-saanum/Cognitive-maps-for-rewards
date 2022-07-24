%#####################################################################
%# Use this to check which decimal and binary code corresponds to parallel port triggers!
%#
%# Requirements: GNU Octave with instrument-control package and Psychtoolbox (for Get/WaitSecs and ListenChar) installed
%#
%# To run, simply press F5 from the editor or type testpport on the command window
%#
%#####################################################################
%# Created by Vincent Ka Ming Cheung on 2017 02 20 
%# at the Max Planck Institute for Human Cognitive and Brain Sciences,
%# Leipzig, Germany
%#
%# This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 Internaional License
%#--- Use at your own risk! While I tried my best, I am not responsible for any errors in the code... ---
%####################################################################

clc % clear the command window
ListenChar(-1); % block keyboard access
WaitSecs(.1); % wait so that KbCheck doesn't immediately end the script
waitingtime = 240; % parallel port listening time in seconds

pkg load instrument-control
pp = parallel("/dev/parport0", 1); 
printf('Waiting for trigger... press any key on the keyboard to stop\n');
fflush(stdout);

now = GetSecs;
while GetSecs - now < waitingtime
    status = pp_stat(pp);    
    if status ~= 126 && status ~= 127 % defaults in status port
		printf([num2str(status) '    ' num2str(bitget(status,8:-1:1)) '    at status port \n']);
		fflush(stdout);
	end

    data = pp_data(pp);    
    if data ~= 255 % default in data port = 11111111 in binary
		printf([num2str(data) '    ' num2str(bitget(data,8:-1:1)) '    at data port \n']);
		fflush(stdout);
	end

    ctrl = pp_ctrl(pp);    
    if ctrl ~= 12 % default in control port
		printf([num2str(ctrl) '    ' num2str(bitget(ctrl,8:-1:1)) '    at control port \n']);
		fflush(stdout);
	end	
	
	
	if KbCheck
		break
	end
end
pp_close(pp);
ListenChar(1); % reenable keyboard access
