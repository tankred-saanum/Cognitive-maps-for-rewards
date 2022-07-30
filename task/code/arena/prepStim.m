function stimuli = prepStim(imageDir)

fnames = get_files(imageDir,'*.png');
for i=1:size(fnames,1)
    stimuli(i).image = imread(deblank(fnames(i,:)));
end
% save it in your stimuli folder