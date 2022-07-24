function [estimate_RDM_ltv,evidenceWeight_ltv]=estimateRDMFromStackOfPartials(stackOfPartialRDMs,monitor)


%% preparations
global stackOfPartialRDMs_ltv;
stackOfPartialRDMs_ltv=vectorizeSimmats(stackOfPartialRDMs);
nPairs=size(stackOfPartialRDMs_ltv,2); % number of dissimilarities = number of item pairs
nDissimPartialMats=size(stackOfPartialRDMs_ltv,3);
nItems=(1+sqrt(1+8*nPairs))/2;

weights=evidenceWeights(stackOfPartialRDMs_ltv); % SNR=1 -> weight=1, SNR=.5 -> weight=.25
 
evidenceWeight_ltv=nansum(weights,3);

if ~exist('monitor','var');
    monitor=false;
end


%% initial estimate: mean
initialEstimate=nanmean(stackOfPartialRDMs_ltv,3);
initialEstimate=initialEstimate/sqrt(nansum(initialEstimate.^2));


%% estimate dissimilarity by iterative alignment to mean
cEstimate=initialEstimate;
iterationChangeSSQ=99999;

% visualize convergence
% figw(900); clf; % for visualization of convergence
% iterationI=1;

tic
while iterationChangeSSQ>1e-8
    cPartialRDM_ltv_normalized=nan(size(stackOfPartialRDMs_ltv));
    % align each partial dissimilarity matrix to the current estimate
    for partialRDMI=1:nDissimPartialMats
        cPartialRDM_ltv=stackOfPartialRDMs_ltv(1,:,partialRDMI);
        nonNaN_LOG=~isnan(cPartialRDM_ltv);
        targetSSQ=sum(cEstimate(nonNaN_LOG).^2);

        stackOfPartialRDMs_ltv_normalized(1,:,partialRDMI)=cPartialRDM_ltv/sqrt(sum(cPartialRDM_ltv(nonNaN_LOG).^2))*sqrt(targetSSQ);
    end
    
    % average the aligned partial dissimilarity matrices to obtain the new estimate
    pEstimate=cEstimate;
    
    % unweighted-mean estimate
    %       cEstimate=nanmean(stackOfPartialRDMs_ltv_normalized,3);
    
    % weighted-mean estimate
    cEstimate=nansum(stackOfPartialRDMs_ltv_normalized.*weights,3)./nansum(weights,3);
    
    cEstimate=cEstimate/sqrt(nansum(cEstimate.^2));
    
    iterationChangeSSQ=nansum((cEstimate-pEstimate).^2);
    
    % visualize convergence
    %     if iterationI<=10
    %         subplot(5,2,iterationI);
    %         % sort
    %         [cEstimate_ltv_sorted,i]=sort(cEstimate(1,:));
    %         stackOfPartialRDMs_ltv_normalized_sorted=stackOfPartialRDMs_ltv_normalized(:,i,:);
    %
    %         % draw
    %         plot(cEstimate_ltv_sorted,'o-','Color','k','MarkerFaceColor','k','MarkerEdgeColor','none','LineWidth',3); hold on;
    %         for partialRDMI=1:nDissimPartialMats
    %             cPartialRDM_ltv=stackOfPartialRDMs_ltv_normalized_sorted(1,:,partialRDMI);
    %             nonNaN_pairIs=find(~isnan(cPartialRDM_ltv));
    %             plot(nonNaN_pairIs,cPartialRDM_ltv(nonNaN_pairIs),'o-','Color',col(partialRDMI,:),'MarkerFaceColor',col(partialRDMI,:),'MarkerEdgeColor','none'); hold on;
    %             %plot(cPartialRDM_ltv,'o-','Color',col(partialRDMI,:),'MarkerFaceColor',col(partialRDMI,:),'MarkerEdgeColor','none'); hold on;
    %         end
    %         plot(cEstimate_ltv_sorted,'-','Color','k','MarkerFaceColor','none','MarkerEdgeColor','none','LineWidth',1); hold on;
    %         xlabel({'\bfitem-pair index','\rm(sorted by dissimilarity according to final estimate)'});
    %         ylabel({'\bfdissimilarity','\rm(black: final estimate,','each color: dissimilarities from one arrangement)'});
    %     end
    %     iterationI=iterationI+1;
end
estimate_RDM_ltv=cEstimate;
toc


%% estimate dissimilarity matrix by optimization
% tic
% estimate_RDM_ltv = fminsearch(@(RDMEstimate) deviationBetweenRDMAndStackOfPartials(RDMEstimate),initialEstimate)
% toc

% tic
% estimate_RDM_ltv = fminsearch(@(RDMEstimate) deviationBetweenRDMAndStackOfPartials(RDMEstimate),estimate_RDM_ltv)
% toc


%% visualization
if ~monitor, return; end

col=nan(nDissimPartialMats,3);
colmap=colorScale([0 0.5 1; 1 0 0],64);
for partialRDMI=1:nDissimPartialMats
    %     col(partialRDMI,:)=randomColor;
    col(partialRDMI,:)=colmap(ceil(partialRDMI/nDissimPartialMats*64),:);
end

% sort according to final estimate of RDM
[estimate_RDM_ltv_sorted,i]=sort(estimate_RDM_ltv(1,:));
stackOfPartialRDMs_ltv_sorted=stackOfPartialRDMs_ltv(:,i,:);

figw(1600); clf;
subplot(2,1,1);
plot(estimate_RDM_ltv_sorted,'o-','Color','k','MarkerFaceColor','k','MarkerEdgeColor','none','LineWidth',3); hold on;
for partialRDMI=1:nDissimPartialMats
    cPartialRDM_ltv=stackOfPartialRDMs_ltv_sorted(1,:,partialRDMI);
    nonNaN_pairIs=find(~isnan(cPartialRDM_ltv));
    plot(nonNaN_pairIs,cPartialRDM_ltv(nonNaN_pairIs),'o-','Color',col(partialRDMI,:),'MarkerFaceColor',col(partialRDMI,:),'MarkerEdgeColor','none'); hold on;
    %plot(cPartialRDM_ltv,'o-','Color',col(partialRDMI,:),'MarkerFaceColor',col(partialRDMI,:),'MarkerEdgeColor','none'); hold on;
end
plot(estimate_RDM_ltv_sorted,'-','Color','k','MarkerFaceColor','none','MarkerEdgeColor','none','LineWidth',1); hold on;
xlabel({'\bfitem-pair index','\rm(sorted by dissimilarity according to final estimate)'});
ylabel({'\bfdissimilarity','\rm(black: final estimate,','each color: dissimilarities from one arrangement)'});
axis([1 nPairs 0 nanmax(stackOfPartialRDMs_ltv_sorted(:))*1.01]);
    
subplot(2,1,2);
plot(estimate_RDM_ltv_sorted,'o-','Color','k','MarkerFaceColor','k','MarkerEdgeColor','none','LineWidth',4); hold on;
mx=0;
for partialRDMI=1:nDissimPartialMats
    cPartialRDM_ltv_sorted=stackOfPartialRDMs_ltv_sorted(1,:,partialRDMI);
    nonNaN_LOG=~isnan(cPartialRDM_ltv_sorted);
    targetSSQ=sum(estimate_RDM_ltv_sorted(nonNaN_LOG).^2);
    cPartialRDM_ltv_sorted_normalized=cPartialRDM_ltv_sorted/sqrt(sum(cPartialRDM_ltv_sorted(nonNaN_LOG).^2))*sqrt(targetSSQ);
    plot(cPartialRDM_ltv_sorted_normalized,'.','Color',col(partialRDMI,:),'MarkerFaceColor',col(partialRDMI,:),'MarkerEdgeColor',col(partialRDMI,:),'MarkerSize',5); hold on;
    mx=max(mx,nanmax(cPartialRDM_ltv_sorted_normalized));
end
plot(estimate_RDM_ltv_sorted,'-','Color','k','MarkerFaceColor','none','MarkerEdgeColor','none','LineWidth',1); hold on;
xlabel({'\bfitem-pair index','\rm(sorted by dissimilarity according to final estimate)'});
ylabel({'\bfdissimilarity','\rm(black: final estimate,','each color: \itscaled-to-fit\rm dissimilarities from one arrangement)'});
axis([1 nPairs 0 mx*1.01]);

pageFigure();
addHeadingAndPrint({'MULTI-ARRANGEMENT',any2str('\fontsize{12}\rmnumber of items = ',nItems,', number of pairs = ',nPairs)},'similarityJudgementData\figures');



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dev=deviationBetweenRDMAndStackOfPartials(cEstimate_RDM_ltv)

%% preparations
global stackOfPartialRDMs_ltv;
nPartialRDMs=size(stackOfPartialRDMs_ltv,3);
nPairs=size(stackOfPartialRDMs_ltv,2);

nData=sum(~isnan(stackOfPartialRDMs_ltv(:)));

% for visualization: sort according to reference RDM
% [cEstimate_RDM_ltv_sorted,i]=sort(cEstimate_RDM_ltv(1,:,1));
% stackOfPartialRDMs_ltv_sorted=stackOfPartialRDMs_ltv(:,i,:);
% figure(1600); clf;


%% compute correlations
dev=nData;
for partialRDMI=1:nPartialRDMs
    cPartialRDM_dissims=stackOfPartialRDMs_ltv(1,:,partialRDMI);
    nonNaN_LOG=~isnan(cPartialRDM_dissims);
    nNonNaN=sum(nonNaN_LOG);
    nData=nData+nNonNaN;

    r_0fix=correlation_0fixed(cEstimate_RDM_ltv(nonNaN_LOG),cPartialRDM_dissims(nonNaN_LOG));

    % visualize
    %     subplot(ceil(nPartialRDMs/2),2,partialRDMI);
    %     plot(cEstimate_RDM_ltv_sorted,'o-','Color','k','MarkerFaceColor','k','MarkerEdgeColor','none'); hold on;
    %     plot(stackOfPartialRDMs_ltv_sorted(1,:,partialRDMI),'o','MarkerFaceColor','r','MarkerEdgeColor','none'); hold on;
    %     title(['r_0fix=',num2str(r_0fix)]);
    
    dev=dev-r_0fix*nNonNaN;
end

dev=dev/nData;


