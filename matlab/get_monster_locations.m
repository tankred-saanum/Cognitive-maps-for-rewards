
subj = 140;
savedir = '';
load([savedir,'sub-',num2str(subj),'\data_',num2str(subj)]);
p_obj = data.mat{2}.data.objPositions*15;
writematrix(p_obj, 'monster_locations.csv');