savedir = '';
subStart = 101;
subStop = 152;

num_trials = 100;
N = (subStop - subStart - 4) * num_trials;
M = 10;
choice_data = zeros(N, M);
%exclude = [136, 137, 138, 121];
exclude = [132, 133, 136, 137, 138, 140, 143];
counter = 1;
counter2 = 1;
trials = 1:100;
num_monsters = 12;
num_drops = 3;
header = {'d2_post_x','d2_post_y','d3_pre_x', 'd3_pre_y', 'd3_post_x', 'd3_post_y'};
for subj = subStart:subStop
    %load([savedir,'sub-',num2str(subj),'\data_',num2str(subj)]);
    if ~ ismember(subj, exclude)
        drop_data = zeros(num_monsters, 2*num_drops);  % get x and y times the number of drop experiments we want
        load([savedir, 'sub-',num2str(subj),'\expt_sub-',num2str(subj),'.mat'])
        d2 = expt.pos.session{1, 2};
        d3 = expt.pos.session{1, 3};
        
        d2_xy = d2.post.positioning;
        d3_xy_pre = d3.pre.positioning;
        d3_xy_post = d3.post.positioning;
        %put the data into the matrix
        drop_data(:, 1:2) = d2_xy;
        drop_data(:, 3:4) = d3_xy_pre;
        drop_data(:, 5:6) = d3_xy_post;
        
        output = [header; num2cell(drop_data)];
        
        T = cell2table(output(2:end,:),'VariableNames',output(1,:));
 
        % Write the table to a CSV file
        
        folderName = ['drop_data_', num2str(subj)];
    
        mkdir (folderName)
        
        writetable(T,[folderName,'\drop_locations.csv']);
        
    end    

    

end
