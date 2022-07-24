function weights=evidenceWeights(arrangementDistances)

% we assume constant placement noise. therefore the dissimilarity SNR is
% proportional to the placement distance in any particular arrangement.
% the evidence weight of a given distance is SNR^2 (optimal filter
% theorem). for two diametrically opposed objects at opposite sides of
% the arena, the distance is 1 and the evidence weight is one. for a 
% distance of .5, the SNR is .5, and the evidence weight is .25. thus it
% takes four of those latter distances to add up to the same evidence 
% as one of the former.
% however, for small distances within the range of the placement noise,
% this evidence weighting rule does not make sense: a distance of 0.1 would
% yield an evidence weight of 0.01, requiring 100 trials to match the
% maximum single-trial evidence weight of 1. for closely placed objects,
% the estimate of the evidence weight itself is unreliable because the
% placement noise dominates. so here we use the heuristic that distances
% below 0.2 (about one fifth of the diameter) should all have the same minimum
% evidence weight of 0.2^2=0.04, requiring about 25 trials to add up to the
% maximum single-trial evidence weight of 1.

%% control variables
lowerLimitOfSingleTrialEvidenceWeight=0.2^2;

%% formula
weights=arrangementDistances.^2;
weights(weights<lowerLimitOfSingleTrialEvidenceWeight)=lowerLimitOfSingleTrialEvidenceWeight;