
savedir = '';
subStart = 101;
subStop = 152;

num_trials = 100;
N = (subStop - subStart - 4) * num_trials;
M = 10;
choice_data = zeros(N, M);
exclude = [136, 137, 138, 121];
counter = 1;
counter2 = 1;
trials = 1:100;
for subj = subStart:subStop
    %load([savedir,'sub-',num2str(subj),'\data_',num2str(subj)]);
    if ~ ismember(subj, exclude)
        tdfread([savedir,'sub-',num2str(subj),'\sub-',num2str(subj), '_ses-3_task-choice_events.tsv']);
        
        % assign subject number
        choice_data(counter:(counter2*num_trials), 1) = subj;
        % assign trial number
        choice_data(counter:(counter2*num_trials), 2) = transpose(trials);
        % assign map
        choice_data(counter:(counter2*num_trials), 3) = map(1:2:length(map));
        % assign option1
        choice_data(counter:(counter2*num_trials), 4) = object_left(1:2:length(object_left));
        % assign option2
        choice_data(counter:(counter2*num_trials), 5) = object_right(1:2:length(object_right));
        %chosen object
        choice_data(counter:(counter2*num_trials), 6) = chosen_object(1:2:length(chosen_object));
        % unchosen object
        choice_data(counter:(counter2*num_trials), 7) = unchosen_object(1:2:length(unchosen_object));
        % chosen value
        choice_data(counter:(counter2*num_trials), 8) = chosen_value(1:2:length(chosen_value));
        % unchosen value
        choice_data(counter:(counter2*num_trials), 9) = unchosen_value(1:2:length(unchosen_value));
        % decision
        choice_data(counter:(counter2*num_trials), 10) = decision(1:2:length(decision));
        
        
        
        
        % update trials
        counter = counter + num_trials;
        counter2 = counter2+1;
        
    end

end
header = {'subj','trial','map', 'option1', 'option2', 'chosen_object','unchosen_object', 'chosen_value', 'unchosen_value', 'decision'};


output = [header; num2cell(choice_data)];

T = cell2table(output(2:end,:),'VariableNames',output(1,:));
 
% Write the table to a CSV file
writetable(T,'choice_data.csv');



