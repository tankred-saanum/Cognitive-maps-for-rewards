function [nVerPan nHorPan]=paneling(nPanels,nHorPanOVERnVerPan)

if ~exist('nHorPanOVERnVerPan','var'),nHorPanOVERnVerPan=.5; end;

nHorPan=ceil(sqrt(nHorPanOVERnVerPan*nPanels));
nVerPan=ceil(nPanels/nHorPan);
