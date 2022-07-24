function [estimate_dissimMat_ltv,simulationResults,story]=simJudgmentByMultiArrangement_circArena_liftTheWeakest(imageData,instructionString,options)
% USAGE
%       estimate_dissimMat_ltv=similarityJudgementByRepeatedArrangement_circularArena_randomExp(imageData,'Please arrange the objects according to their similarity.')
%
% FUNCTION
%       This function lets a subject arrange the objects represented by the
%       icon images in the argument imageData in a circular "arena" by
%       dragging and dropping with the mouse. The subject is first
%       presented with the entire set and then with subsets (using the
%       "lift the weakest" heuristic for subset construction). The function
%       returns a single dissimilarity matrix that represents the subject's
%       judgements. This matrix is estimated by combining repeated
%       judgements on each given pair of objects.




%% set unset options
if ~exist('options','var')||isempty(options), options=struct; end
options=setIfUnset(options,'subjectInitials','XX');
options=setIfUnset(options,'sessionI',0);
options=setIfUnset(options,'axisUnits','Pixels');
options=setIfUnset(options,'maxSessionLength_min',inf);
options=setIfUnset(options,'analysisFigs',true);
options=setIfUnset(options,'saveEachTrial',true);


%% control variables
evidenceUtilityExponent=10;
minRequiredEvidenceWeight=.5;


%% preparations
nObjects=numel(imageData);
nPairs=(nObjects^2-nObjects)/2;


%% simulation mode
simulationMode=false;
if simulationMode
    % create distance matrix from random full-dimensionality gaussian
    % arrangement
    simulatedTrueObjectPositions=randn(nObjects,nObjects-1);
    simulatedTrueDistMat_ltv=pdist(simulatedTrueObjectPositions,'euclidean');
    simulatedTrueDistMat=squareform(simulatedTrueDistMat_ltv);
end


%% trial object-set construction by "lift the weakest" heuristic

% show evidence utility as a function of evidence weight
w=0:.001:2;

if options.analysisFigs
    figw(100); clf;
    plot(w,1-exp(-w*evidenceUtilityExponent),'LineWidth',3);
    xlabel('evidence weight');
    ylabel('evidence utility');
end

% initialization
subjectWork_nObjectsArranged=0;
subjectWork_nPairsArranged=0;
subjectWork_nDragsEstimate=0;

minEvidenceWeight=0;
cTrial_objectIs=1:nObjects;
distMatsForAllTrials_ltv=[];

% prepare conversion from lower-triangular-vector to square indices
[verIs,horIs]=ndgrid(1:nObjects,1:nObjects);
verIs_ltv=vectorizeSimmat(verIs);
horIs_ltv=vectorizeSimmat(horIs);

% prepare conversion from square to lower-triangular-vector indices
ltvIs_ltv=1:numel(horIs_ltv);
ltvIs_sq=squareSimmat(ltvIs_ltv);


%% multiple arrangement trials
trialI=0;
while minEvidenceWeight<minRequiredEvidenceWeight &&... 
      (trialI==0||etime(clock,trialStartTimes(1,:))<options.maxSessionLength_min*60) % trial loop
    
    % perfrom the next trial: let subject arrange a set of objects 
    trialI=trialI+1;
    trialStartTimes(trialI,:)=clock; % jot down trial beginning time
    if simulationMode
        objectPositions=mdscale(simulatedTrueDistMat(cTrial_objectIs,cTrial_objectIs),2,'Criterion','metricstress');
        distMat_ltv=pdist(objectPositions,'euclidean');

        % scale to fill arena (maximum distance = 1)
        objectPositions=objectPositions/max(distMat_ltv);

        % add noise (simulating slight misplacement by the subject)
        noiseStd=.1; % vertical and horizontal gaussian noise standard deviation in arena-diameter units
        objectPositions=objectPositions+randn(size(objectPositions))*noiseStd;

        distMat_ltv=pdist(objectPositions,'euclidean');
        distMat_ltv=distMat_ltv/max(distMat_ltv); % confine to arena again after noise application
    else
        [objectPositions,distMat_ltv]=letSubjectArrangeItems_circularArena(imageData(cTrial_objectIs),instructionString,options);
    end
    trialStopTimes(trialI,:)=clock; % jot down trial termination time
    trialDurations(trialI)=etime(trialStopTimes(trialI,:),trialStartTimes(trialI,:));
    
    % keep track of subject work
    nObjectsToBeArranged=numel(cTrial_objectIs);
    subjectWork_nObjectsArranged=subjectWork_nObjectsArranged+nObjectsToBeArranged;
    subjectWork_nPairsArranged=subjectWork_nPairsArranged+(nObjectsToBeArranged^2-nObjectsToBeArranged)/2;
    subjectWork_nDragsEstimate=subjectWork_nDragsEstimate+sqrt((nObjectsToBeArranged^2-nObjectsToBeArranged)/2)^1.5;

    % include completed trial evidence in distMatsForAllTrials_ltv
    distMatFullSize=nan(nObjects);
    distMatFullSize(cTrial_objectIs,cTrial_objectIs)=squareform(distMat_ltv,'tomatrix');
    distMatFullSize_ltv=vectorizeSimmat(distMatFullSize);
    distMatsForAllTrials_ltv=cat(3,distMatsForAllTrials_ltv,distMatFullSize_ltv);

    % estimate the dissimilarity matrix from the current evidence
    % (and the current evidence weight for each pair)
    [estimate_dissimMat_ltv,evidenceWeight_ltv]=estimateDissimMatFromStackOfPartials(distMatsForAllTrials_ltv,options.analysisFigs);
    evidenceWeight_sq=squareform(evidenceWeight_ltv);
    
    % visualize current dissimilarity matrix estimate (and evidence weights) 
    if options.analysisFigs
        showSimmats(estimate_dissimMat_ltv,110); title('current dissimilarity matrix estimate');
        [estimate_dissimMat_ltv_sorted,sortingIs]=sort(estimate_dissimMat_ltv); % sort according to current estimate of dissimMat
        figw(120); clf; subplot(2,1,1);
        image(evidenceWeight_sq,'CDataMapping','scaled'); colorbar; title('\bfcurrent evidence weight');
        caxis([0 max(evidenceWeight_ltv)]); axis square;
        subplot(2,1,2); plot(evidenceWeight_ltv(sortingIs),'o','MarkerFaceColor','k','MarkerEdgeColor','none');
        line([1 nPairs],[minRequiredEvidenceWeight minRequiredEvidenceWeight],'Color',[.6 .6 .6],'LineWidth',3);
        axis([1 nPairs 0 2]);
        xlabel({'\bfobject-pair index','\rm(sorted by dissimilarity according to current estimate)'});
        ylabel('\bfcurrent evidence weight');
    end

    % estimate the current evidence utility for each pair
    evidenceUtility_ltv=1-exp(-evidenceWeight_ltv*evidenceUtilityExponent);
    evidenceUtility_sq=squareform(evidenceUtility_ltv);

    % save this trial's information
    if options.saveEachTrial
        save([options.subjectInitials,'_session',num2str(options.sessionI),'_trial',num2str(trialI)],...
            'trialStartTimes','trialStopTimes','trialDurations','objectPositions','distMat_ltv',...
            'nObjectsToBeArranged',...
            'subjectWork_nObjectsArranged','subjectWork_nPairsArranged','subjectWork_nDragsEstimate',...
            'distMatFullSize_ltv','distMatsForAllTrials_ltv',...
            'estimate_dissimMat_ltv',...
            'evidenceWeight_ltv','evidenceUtility_ltv');
    end

    % choose first pair to include
    [minVal,minI]=min(evidenceUtility_ltv(:));
    object1I=verIs_ltv(minI);
    object2I=horIs_ltv(minI);

    % re-initialize current-trial object set
    cTrial_objectIs=[object1I,object2I];

    % consider adding another object
    while true
        trialEfficiencies=nan(nObjects-numel(cTrial_objectIs)+1,1);

        otherObjectIs=setdiff(1:nObjects, cTrial_objectIs);
        objectAddedI=[];
        objectSetI=1;

        % consider each other object
        while true % object loop
            % compute trial utility
            estimate_dissimMat_sq=squareSimmat(estimate_dissimMat_ltv);
            estimate_dissimMat_sq_cTrial=estimate_dissimMat_sq(cTrial_objectIs,cTrial_objectIs);

            if max(estimate_dissimMat_sq_cTrial(:))>0 % if partial RDM can be aligned
                estimate_dissimMat_sq_cTrial=estimate_dissimMat_sq_cTrial/max(estimate_dissimMat_sq_cTrial(:)); % scale to peak at 1

                utilityBeforeTrial=sum(squareform(evidenceUtility_sq(cTrial_objectIs,cTrial_objectIs)));
                evidenceWeightAfterTrial_sq=evidenceWeight_sq(cTrial_objectIs,cTrial_objectIs)+evidenceWeights(estimate_dissimMat_sq_cTrial);
                evidenceWeightAfterTrial_sq(logical(eye(numel(cTrial_objectIs))))=0;
                utilityAfterTrial=sum(1-exp(-squareform(evidenceWeightAfterTrial_sq)*evidenceUtilityExponent));

                utilityBenefit=utilityAfterTrial-utilityBeforeTrial;
            else
                utilityBenefit=0; % partial RDM couldn't be aligned (all 0) -> cannot estimate trial utility -> assume unuseful trial
            end
            %trialCost=numel(cTrial_objectIs); % number of objects (minimum: an underestimate)
            %trialCost=numel(cTrial_objectIs)^2; % number of objects (maximum: an overestimate)
            trialCost=numel(cTrial_objectIs)^1.5; % number of objects (maximum: an overestimate)

            trialEfficiencies(objectSetI)=utilityBenefit/trialCost

            cTrial_objectIs=setdiff(cTrial_objectIs,objectAddedI); % take out the previously added object (none on the first iteration)

            if objectSetI==numel(otherObjectIs)+1,
                break;
            end

            objectAddedI=otherObjectIs(objectSetI);
            cTrial_objectIs=union(cTrial_objectIs,objectAddedI);
            objectSetI=objectSetI+1;
        end % object loop

        [maxVal,maxI]=max(trialEfficiencies);

        if maxI==1
            if numel(cTrial_objectIs)>=3
                % do not add more objects
                break;
            else
                % pair trial has greatest efficiency,
                % but for scale-invariant estimation we need at least
                % 3 objects in a trial. add the one that renders the trial
                % most efficient.
                [maxVal,maxI]=max(trialEfficiencies(2:end));
                cTrial_objectIs=union(cTrial_objectIs,otherObjectIs(maxI));
            end
        else
            % add the object bringing the greatest utility gain
            cTrial_objectIs=union(cTrial_objectIs,otherObjectIs(maxI-1));
        end
    end % object-set definition for next trial

    minEvidenceWeight=min(evidenceWeight_ltv);
    
end % trial loop 

                % displayStackOfPartialEvidenceDistanceMats(distMatsForAllTrials);
                % save('distMatsForAllTrials','distMatsForAllTrials');

%% return dissimilarity estimate (and trial times and arrangements)                
[estimate_dissimMat_ltv,evidenceWeight_ltv]=estimateDissimMatFromStackOfPartials(distMatsForAllTrials_ltv,options.analysisFigs);
story.distMatsForAllTrials_ltv=distMatsForAllTrials_ltv;
story.trialStartTimes=trialStartTimes;
story.trialStopTimes=trialStopTimes;
story.trialDurations=trialDurations;
story.evidenceWeight_ltv=evidenceWeight_ltv;

% save all inputs and outputs (except simulation results)
save([options.subjectInitials,'_session',num2str(options.sessionI),'_',dateAndTimeStringNoSec_nk],...
    'imageData','instructionString','options','estimate_dissimMat_ltv','evidenceWeight_ltv','story');


%% show simulation results
if simulationMode
    figw(1700); clf;
    
    % display relationship of estimated and true dissimilarities for
    % the multi-arrangement method
    subplot(2,1,1);
    plot(simulatedTrueDistMat_ltv,estimate_dissimMat_ltv,'o','MarkerFaceColor','k','MarkerEdgeColor','none');
    xlabel('simulated true dissimilarity');
    ylabel('dissimilarity estimated from multiarrangement');
    title({'\bf MULTI-ARRANGEMENT (lift the weakest): correlation(true, estimated dissim.)',...
           any2str('r=',corr(simulatedTrueDistMat_ltv(:),estimate_dissimMat_ltv(:)),...
           '\rm (misplacement noise=',noiseStd,', #drags=',subjectWork_nObjectsArranged,...
           ', #drags/#pairs=',subjectWork_nObjectsArranged/nPairs)});
    axis square;
    
    % display relationship of estimated and true dissimilarities for
    % the pairwise absolute similarity maesurement
    pairwiseDissimMeasures_ltv=simulatedTrueDistMat_ltv/max(simulatedTrueDistMat_ltv);
    pairwiseDissimMeasures_ltv=pairwiseDissimMeasures_ltv+randn(size(pairwiseDissimMeasures_ltv))*noiseStd;

    subplot(2,1,2);
    plot(simulatedTrueDistMat_ltv,pairwiseDissimMeasures_ltv,'o','MarkerFaceColor','k','MarkerEdgeColor','none');
    xlabel('simulated true dissimilarity');
    ylabel('dissimilarity estimated from pairwise judgement');
    title({'\bf PAIRWISE JUDGMENT: correlation(true, estimated dissim.)',...
           any2str('r=',corr(simulatedTrueDistMat_ltv(:),pairwiseDissimMeasures_ltv(:)),...
           '\rm (misplacement noise=',noiseStd,', #drags=#pairs=',nPairs)});
    axis square;

    pageFigure();
    addHeadingAndPrint(any2str('number of objects = ',nObjects),'figures');
    
end


%% return simulation results
if simulationMode
    simulationResults.r_trueEstimated_multiArangement=corr(simulatedTrueDistMat_ltv(:),estimate_dissimMat_ltv(:));
    simulationResults.r_trueEstimated_pairwiseArangement=corr(simulatedTrueDistMat_ltv(:),pairwiseDissimMeasures_ltv(:));
    simulationResults.nPairs=nPairs;
    simulationResults.nObjects=nObjects;
    simulationResults.subjectWork_nObjectsArranged=subjectWork_nObjectsArranged;
    simulationResults.subjectWork_nPairsArranged=subjectWork_nPairsArranged;
    simulationResults.subjectWork_nDragsEstimate=subjectWork_nDragsEstimate;
else
    simulationResults=[];
end


%% revert to original directory
% cd('..');
