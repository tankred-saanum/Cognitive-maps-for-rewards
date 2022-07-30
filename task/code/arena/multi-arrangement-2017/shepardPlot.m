function shepardPlot(dissimilarities,disparities,distances,figI,titleString)
% displays a Shepard plot for the result of multidimensional scaling (MDS)
%
% dissimilarities
%           the dissimilarities to be depicted
% [disparities]
%           the corresponding monotonically transformed distances MDS
%           strives to produce in the low-dimensional arrangement
% distances
%           the actual distances of the low-dimensional MDS arrangement


%% preparations
if ~exist('figI','var'), figI=0; end
selectPlot(figI); cla; hold on;

dissimilarities=vectorizeSimmats(dissimilarities);
disparities=vectorizeSimmats(disparities);
distances=vectorizeSimmats(distances);


%% plotting

% mx_dissim=max(dissimilarities);
% plot([0 mx_dissim],[0 mx_dissim],'k-'); 

plot(dissimilarities,distances,'bo');

if ~isempty(disparities)
    disparities=vectorizeSimmat(disparities);
    [dum,ord] = sortrows([disparities(:) dissimilarities(:)]);
    plot(dissimilarities(ord),disparities(ord),'r.-');
end


%% labels
xlabel('dissimilarity'); ylabel('distance (blue), disparity (red)')
legend({'distance' 'disparities' '1:1 line'},'Location','NorthWest');

r_Pearson=corr(dissimilarities(:),distances(:),'type','Pearson');
r_Spearman=corr(dissimilarities(:),distances(:),'type','Spearman');
corrString=['corr(dissimilarity,distance)=',num2str(r_Pearson,2),' (Pearson), =',num2str(r_Spearman,2),' (Spearman)'];

if ~exist('titleString','var')
    title('\fontsize{14}Shepard plot\fontsize{9}',corrString);
else
    title({'\fontsize{14}Shepard plot\fontsize{9}',titleString,corrString});
end