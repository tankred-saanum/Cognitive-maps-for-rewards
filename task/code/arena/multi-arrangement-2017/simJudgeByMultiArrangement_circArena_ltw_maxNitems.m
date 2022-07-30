function [estimate_RDM_ltv,simulationResults,story]=simJudgeByMultiArrangement_circArena_ltw_maxNitems(imageData,instructionString,options)
% USAGE
%       [estimate_RDM_ltv,simulationResults,story] = ...
%            simJudgeByMultiArrangement_circArena_ltw_maxNitems(imageData,in
%            structionString,options)
%
% FUNCTION
%       This function lets a subject arrange the items represented by the
%       icon images in the argument imageData in a circular "arena" by
%       dragging and dropping with the mouse. The item sets to be
%       presented in each trial are adaptively designed using the "lift the
%       weakest" heuristic for subset construction. The function returns a
%       single dissimilarity matrix that represents the subject's
%       judgements. This matrix is estimated by combining repeated
%       judgements on each given pair of items.
%
% Nikolaus Kriegeskorte & Marieke Mur
% original version: 2008
% this version: April 2015 (control maximum number of stimuli per trial using options.maxNitemsPerTrial)
%
% citation: Kriegeskorte N, Mur M (2012) Inverse MDS: inferring dissimilarity structure from multiple item arrangements. Frontiers in psychology, 3.


%% set unset options
if ~exist('options','var')||isempty(options), options=struct; end
options=setIfUnset(options,'subjectInitials','XX');
options=setIfUnset(options,'sessionI',0);
options=setIfUnset(options,'axisUnits','Pixels');
options=setIfUnset(options,'maxSessionLength_min',inf);
options=setIfUnset(options,'analysisFigs',false);
options=setIfUnset(options,'saveEachTrial',true);
options=setIfUnset(options,'simulationMode',false);
options=setIfUnset(options,'maxNitemsPerTrial',inf);


%% control variables
evidenceUtilityExponent=10;
minRequiredEvidenceWeight=.5;
dragsExponent=1.2;


%% preparations
nItems=numel(imageData);
nPairs=(nItems^2-nItems)/2;


%% simulation mode
if options.simulationMode
    % create distance matrix from random full-dimensionality gaussian
    % arrangement
    simulatedTrueItemPositions=randn(nItems,nItems-1);
    simulatedTrueDistMat_ltv=pdist(simulatedTrueItemPositions,'euclidean');
    simulatedTrueDistMat=squareform(simulatedTrueDistMat_ltv);
end


%% trial item-set construction by "lift the weakest" heuristic

% show evidence utility as a function of evidence weight
w=0:.001:2;

if options.analysisFigs
    figw(100); clf;
    subplot(3,3,5);
    plot(w,1-exp(-w*evidenceUtilityExponent),'LineWidth',3);
    xlabel('evidence weight');
    ylabel('evidence utility');
    
    pageFigure();
    if options.simulationMode
        mode='arrangements simulated by MDS (for random ground-truth dissimilarity matrix)';
    else
        mode='arrangements by human subject';
    end
    
    addHeadingAndPrint({'MULTI-ARRANGEMENT RUN DOCUMENTATION',any2str('\rmnumber of items = ',nItems,', number of pairs = ',nPairs),any2str('max number of items per trial = ',options.maxNitemsPerTrial),mode},'similarityJudgementData\figures');

end

% prepare conversion from lower-triangular-vector to square indices
[verIs,horIs]=ndgrid(1:nItems,1:nItems);
verIs_ltv=vectorizeSimmat(verIs);
horIs_ltv=vectorizeSimmat(horIs);

% prepare conversion from square to lower-triangular-vector indices
ltvIs_ltv=1:numel(horIs_ltv);
%ltvIs_sq=squareSimmat(ltvIs_ltv);

% initialization
subjectWork_nItemsArranged=0;
subjectWork_nPairsArranged=0;
subjectWork_nDragsEstimate=0;

minEvidenceWeight=0;
evidenceWeight_ltv=zeros(size(verIs_ltv)); % before the first trial, there is no evidence.
%cTrial_itemIs=1:nItems;
distMatsForAllTrials_ltv=[];

weakestEvidenceWeights=[];
meanEvidenceWeights=[];


%% multiple arrangement trials
trialI=0;
while minEvidenceWeight<minRequiredEvidenceWeight &&... 
      (any(evidenceWeight_ltv==0)||etime(clock,trialStartTimes(1,:))<options.maxSessionLength_min*60) % trial loop

  
    % estimate the current evidence utility for each pair
    evidenceUtility_ltv=1-exp(-evidenceWeight_ltv*evidenceUtilityExponent);
    evidenceUtility_sq=squareform(evidenceUtility_ltv);
  
    % choose first pair to include 
    % (to ensure that the trial can be aligned to previous trials, this
    % pair is chosen randomly the ones for which we already have some
    % evidence...
    evidenceLOG_ltv=evidenceUtility_ltv>0;
    
    % ...among these we choose the one whose two items, paired
    % with other items, reach the largest number of zero-evidence pairs.)
    
    if any(evidenceLOG_ltv)
        % there are pairs we already have evidence for.
        evidenceUtility_sq_nan=evidenceUtility_sq;
        evidenceUtility_sq_nan(logical(eye(nItems)))=nan;
        nObjEachObjHasNotBeenPairedWith = nansum(evidenceUtility_sq_nan==0);
        nZeroEvidencePairsReachedByEachPair=repmat(nObjEachObjHasNotBeenPairedWith',[1 nItems])+repmat(nObjEachObjHasNotBeenPairedWith,[nItems 1]);
        nZeroEvidencePairsReachedByEachPair(logical(eye(nItems)))=0;
        nZeroEvidencePairsReachedByEachPair_ltv=squareform(nZeroEvidencePairsReachedByEachPair);
        nZeroEvidencePairsReachedByEachPair_ltv(~evidenceLOG_ltv)=0;
        [maxVal,maxI] = max(nZeroEvidencePairsReachedByEachPair_ltv);
        maxIs=find(nZeroEvidencePairsReachedByEachPair_ltv==maxVal);
        maxI=maxIs(ceil(rand*numel(maxIs)));
    else
        % there are no pairs we already have evidence for.
        initialPairI=ceil(rand*nPairs); % choose random pair
    end
    
    item1I=verIs_ltv(initialPairI);
    item2I=horIs_ltv(initialPairI);

    % re-initialize current-trial item set
    cTrial_itemIs=[item1I,item2I];

    % consider adding additional items
    while numel(cTrial_itemIs)<options.maxNitemsPerTrial
        trialEfficiencies=nan(nItems-numel(cTrial_itemIs)+1,1);

        otherItemIs=setdiff(1:nItems, cTrial_itemIs);
        itemAddedI=[];
        itemSetI=1;

        % consider each other item
        while true % item loop
            % compute trial utility
            if exist('estimate_RDM_ltv')
                estimate_RDM_sq = squareSimmat(estimate_RDM_ltv);
            else
                estimate_RDM_sq = ones(nItems,nItems);
                estimate_RDM_sq(logical(eye(nItems)))=0;
                estimate_RDM_ltv=squareform(estimate_RDM_sq);
            end
            estimate_RDM_sq_cTrial = estimate_RDM_sq(cTrial_itemIs,cTrial_itemIs);

            if max(estimate_RDM_sq_cTrial(:))>0 % if partial RDM can be aligned

                % if the currently considered trial contains
                % dissimilarities that we have no evidence for, assume that
                % they equal the median of the dissimilarities we do have
                % evidence for in order to estimate the trial efficiency.
                estimate_RDM_sq_cTrial(isnan(estimate_RDM_sq_cTrial))=median(estimate_RDM_ltv(find(~isnan(estimate_RDM_ltv))));
                estimate_RDM_sq_cTrial=estimate_RDM_sq_cTrial/max(estimate_RDM_sq_cTrial(:)); % scale to peak at 1
                
                utilityBeforeTrial=sum(squareform(evidenceUtility_sq(cTrial_itemIs,cTrial_itemIs)));
                evidenceWeight_sq=squareform(evidenceWeight_ltv);
                evidenceWeightAfterTrial_sq=evidenceWeight_sq(cTrial_itemIs,cTrial_itemIs)+evidenceWeights(estimate_RDM_sq_cTrial);
                evidenceWeightAfterTrial_sq(logical(eye(numel(cTrial_itemIs))))=0;
                utilityAfterTrial=sum(1-exp(-squareform(evidenceWeightAfterTrial_sq)*evidenceUtilityExponent));

                utilityBenefit=utilityAfterTrial-utilityBeforeTrial;
            else
                utilityBenefit=0; % partial RDM couldn't be aligned (all 0) -> cannot estimate trial utility -> assume unuseful trial
            end
            %trialCost=numel(cTrial_itemIs); % number of items (minimum: an underestimate)
            %trialCost=numel(cTrial_itemIs)^2; % number of items^2 (maximum: an overestimate)
            trialCost=numel(cTrial_itemIs)^dragsExponent; % number of items^dragsExponent (our estimate)

            trialEfficiencies(itemSetI)=utilityBenefit/trialCost;

            cTrial_itemIs=setdiff(cTrial_itemIs,itemAddedI); % take out the previously added item (none on the first iteration)

            if itemSetI==numel(otherItemIs)+1,
                break;
            end

            itemAddedI=otherItemIs(itemSetI);
            cTrial_itemIs=union(cTrial_itemIs,itemAddedI);
            itemSetI=itemSetI+1;
        end % item loop

        [maxVal,maxI]=max(trialEfficiencies);
        maxIs=find(trialEfficiencies==maxVal);
        maxI=maxIs(ceil(rand*numel(maxIs)));

        if maxI==1
            % adding another item would not improve trial efficiency
            if numel(cTrial_itemIs)>=3
                % do not add more items
                break;
            else
                % pair trial has greatest efficiency,
                % but for scale-invariant estimation we need at least
                % 3 items in a trial. add the one that renders the trial
                % most efficient.
                [maxVal,maxI]=max(trialEfficiencies(2:end));
                maxIs=find(trialEfficiencies(2:end)==maxVal);
                maxI=maxIs(ceil(rand*numel(maxIs)));
                cTrial_itemIs=union(cTrial_itemIs,otherItemIs(maxI));
            end
        else
            % add the item bringing the greatest utility gain
            cTrial_itemIs=union(cTrial_itemIs,otherItemIs(maxI-1));
        end
    end % item-set definition for next trial
  
  
    % perfrom the next trial: let subject arrange a set of items 
    trialI=trialI+1;
    trialStartTimes(trialI,:)=clock; % jot down trial beginning time
    if options.simulationMode
        itemPositions=mdscale(simulatedTrueDistMat(cTrial_itemIs,cTrial_itemIs),2,'Criterion','metricstress');
        distMat_ltv=pdist(itemPositions,'euclidean');

        % scale to fill arena (maximum distance = 1)
        itemPositions=itemPositions/max(distMat_ltv);

        % add noise (simulating slight misplacement by the subject)
        noiseStd=.1; % vertical and horizontal gaussian noise standard deviation in arena-diameter units
        itemPositions=itemPositions+randn(size(itemPositions))*noiseStd;

        distMat_ltv=pdist(itemPositions,'euclidean');
        distMat_ltv=distMat_ltv/max(distMat_ltv); % confine to arena again after noise application
    else
        [itemPositions,distMat_ltv]=letSubjectArrangeItems_circularArena(imageData(cTrial_itemIs),{instructionString,'\rm(left mouse button to drag, right mouse button to multiselect by clicking or dragging, A & Z to zoom)'},options);
    end
    trialStopTimes(trialI,:)=clock; % jot down trial termination time
    trialDurations(trialI)=etime(trialStopTimes(trialI,:),trialStartTimes(trialI,:));
    
    % keep track of subject work
    nsItemsPerTrial(trialI)=numel(cTrial_itemIs);
    subjectWork_nItemsArranged=subjectWork_nItemsArranged+nsItemsPerTrial(trialI);
    subjectWork_nPairsArranged=subjectWork_nPairsArranged+(nsItemsPerTrial(trialI)^2-nsItemsPerTrial(trialI))/2;
    subjectWork_nDragsEstimate=subjectWork_nDragsEstimate+sqrt((nsItemsPerTrial(trialI)^2-nsItemsPerTrial(trialI))/2)^dragsExponent;

    % include completed trial evidence in distMatsForAllTrials_ltv
    distMatFullSize=nan(nItems);
    distMatFullSize(cTrial_itemIs,cTrial_itemIs)=squareform(distMat_ltv,'tomatrix');
    distMatFullSize_ltv=vectorizeSimmat(distMatFullSize);
    distMatsForAllTrials_ltv=cat(3,distMatsForAllTrials_ltv,distMatFullSize_ltv);

    % estimate the dissimilarity matrix from the current evidence
    % (and the current evidence weight for each pair)
    [estimate_RDM_ltv,evidenceWeight_ltv]=estimateRDMFromStackOfPartials(distMatsForAllTrials_ltv,options.analysisFigs);
    evidenceWeight_sq=squareform(evidenceWeight_ltv);
    
    weakestEvidenceWeights(trialI)=min(evidenceWeight_ltv);
    meanEvidenceWeights(trialI)=mean(evidenceWeight_ltv); 
    
    % visualize current dissimilarity matrix estimate (and evidence weights) 
    if options.analysisFigs
        showRDMs(estimate_RDM_ltv,110); title('current dissimilarity matrix estimate');
        [estimate_RDM_ltv_sorted,sortingIs]=sort(estimate_RDM_ltv); % sort according to current estimate of RDM
        figw(120); clf; subplot(4,1,1);
        image(evidenceWeight_sq,'CDataMapping','scaled'); colormap('bone'); colorbar; title('\bfcurrent evidence weight');
        caxis([0 max(evidenceWeight_ltv)]); axis square;
        subplot(4,1,2); plot(evidenceWeight_ltv(sortingIs),'o','MarkerFaceColor','k','MarkerEdgeColor','none');
        line([1 nPairs],[minRequiredEvidenceWeight minRequiredEvidenceWeight],'Color',[.6 .6 .6],'LineWidth',3);
        axis([1 nPairs 0 2]);
        xlabel({'\bfitem-pair index','\rm(sorted by dissimilarity according to current estimate)'});
        ylabel('\bfcurrent evidence weight');
        subplot(4,1,3); plot(nsItemsPerTrial,'o','MarkerFaceColor','k','MarkerEdgeColor','none');
        xlabel('\bftrial index');
        ylabel('\bfnumber or items');
        subplot(4,1,4); hold on;
        plot(weakestEvidenceWeights,'-','LineWidth',5,'Color',[0.5 0.5 0.5]);
        plot(meanEvidenceWeights,'-','LineWidth',3,'Color','k');
        xlabel('\bftrial index');
        ylabel({'\bfmin across pairs of the evidence (gray)','mean evidence (black)'});
        
        if  options.simulationMode
            % to estimate how well a partial dissimilarity matrix
            % represents the true dissimilarity (ground truth known here as
            % this is simulation mode), let's assume we estimate the unknown
            % dissimilarities to be the median of the known ones.
            estimate_RDM_ltv_nan2med=estimate_RDM_ltv;
            estimate_RDM_ltv_nan2med(isnan(estimate_RDM_ltv))=median(estimate_RDM_ltv(~isnan(estimate_RDM_ltv)));
            accuracyOfDissimEstimates(trialI)= corr(simulatedTrueDistMat_ltv(:),estimate_RDM_ltv_nan2med(:));
            plot(accuracyOfDissimEstimates,'-','LineWidth',2,'Color','r');
            ylabel({'\bfmin across pairs of the evidence (gray)','mean evidence (black)','corr(dissim est, ground truth) (red)'});
        end
    end

    % save this trial's information
    if options.saveEachTrial
        save(['similarityJudgementData\',num2str(options.subjectID),'_',options.subjectInitials,'_session',num2str(options.sessionI),'_trial',num2str(trialI)],...
            'trialStartTimes','trialStopTimes','trialDurations','itemPositions','distMat_ltv',...
            'cTrial_itemIs','nsItemsPerTrial',...
            'subjectWork_nItemsArranged','subjectWork_nPairsArranged','subjectWork_nDragsEstimate',...
            'distMatFullSize_ltv','distMatsForAllTrials_ltv',...
            'estimate_RDM_ltv',...
            'evidenceWeight_ltv','evidenceUtility_ltv');
    end

    minEvidenceWeight=min(evidenceWeight_ltv);
    
end % trial loop 

% displayStackOfPartialEvidenceDistanceMats(distMatsForAllTrials);
% save('distMatsForAllTrials','distMatsForAllTrials');


%% return dissimilarity estimate (and trial times and arrangements)                
[estimate_RDM_ltv,evidenceWeight_ltv]=estimateRDMFromStackOfPartials(distMatsForAllTrials_ltv,options.analysisFigs);
story.distMatsForAllTrials_ltv=distMatsForAllTrials_ltv;
story.trialStartTimes=trialStartTimes;
story.trialStopTimes=trialStopTimes;
story.trialDurations=trialDurations;
story.evidenceWeight_ltv=evidenceWeight_ltv;
story.nsItemsPerTrial=nsItemsPerTrial;

% save all inputs and outputs (except simulation results)
% save(['similarityJudgementData\',options.subjectInitials,'_session',num2str(options.sessionI),'_',dateAndTimeStringNoSec_nk],...
%     'imageData','instructionString','options','estimate_RDM_ltv','evidenceWeight_ltv','story');

% pageFigure(120);
% addHeadingAndPrint(any2str('number of items = ',nItems),'similarityJudgementData\figures');


%% show simulation results
if options.simulationMode
    figw(1700); clf;
    
    % display relationship of estimated and true dissimilarities for
    % the multi-arrangement method
    subplot(2,1,1);
    plot(simulatedTrueDistMat_ltv,estimate_RDM_ltv,'o','MarkerFaceColor','k','MarkerEdgeColor','none');
    xlabel('simulated true dissimilarity');
    ylabel('dissimilarity estimated from multiarrangement');
    title({'\bf MULTI-ARRANGEMENT (lift the weakest): correlation(true, estimated dissim.)',...
           any2str('r=',corr(simulatedTrueDistMat_ltv(:),estimate_RDM_ltv(:)),...
           '\rm (placement noise sigma = ',noiseStd,', total # items placed = ',subjectWork_nItemsArranged,...
           ', # obj placed / # pairs = ',subjectWork_nItemsArranged/nPairs)});
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
           '\rm (placement noise sigma = ',noiseStd,', # pairs judged = ',nPairs)});
    axis square;

    pageFigure();
    addHeadingAndPrint(any2str('number of items = ',nItems),'similarityJudgementData\figures');

    
end
     
    
%% return simulation results
if options.simulationMode
    simulationResults.r_trueEstimated_multiArangement=corr(simulatedTrueDistMat_ltv(:),estimate_RDM_ltv(:));
    simulationResults.r_trueEstimated_pairwiseArangement=corr(simulatedTrueDistMat_ltv(:),pairwiseDissimMeasures_ltv(:));
    simulationResults.nPairs=nPairs;
    simulationResults.nItems=nItems;
    simulationResults.subjectWork_nItemsArranged=subjectWork_nItemsArranged;
    simulationResults.subjectWork_nPairsArranged=subjectWork_nPairsArranged;
    simulationResults.subjectWork_nDragsEstimate=subjectWork_nDragsEstimate;
else
    simulationResults=[];
end


