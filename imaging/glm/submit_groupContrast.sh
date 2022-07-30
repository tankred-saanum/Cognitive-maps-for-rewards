#!/bin/bash -l

# (c) Mona Garvert <garvert@cbs.mpg.de>  August 2019

# needs to run in the matlab environment!

for des in 118 119 120;
do
	cat /data/p_02071/choice-maps/scripts/designs/groupContrasts.m | sed "s/XXdesignXX/${des}/g" > /data/p_02071/choice-maps/scripts/designs/groupContrasts2.m 
	
			condor_submit /data/p_02071/choice-maps/scripts/designs/run_groupContrasts
done
