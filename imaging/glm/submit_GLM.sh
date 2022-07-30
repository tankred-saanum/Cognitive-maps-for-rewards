#!/bin/bash -l

# (c) Mona Garvert <garvert@cbs.mpg.de>  August 2019

# needs to run in the matlab environment!

for des in 15781;
do
		
	for subj in {101..152};do
    	for sess in 2 3;do

			echo $subj
			echo $sess
		
			mkdir -p /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/	
				
			# design and wrapper scripts
			cat /data/p_02071/choice-maps/scripts/designs/design_${des}.m | sed "s/XXsubjIDXX/${subj}/g" | sed "s/XXsessionXX/${sess}/g" | sed "s/XXdesignXX/${des}/g" > /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/run_GLM.m
			cat /data/p_02071/choice-maps/scripts/designs/run_glm.sh | sed "s/SUBJID/${subj}/g"  | sed "s/SESSID/${sess}/g" | sed "s/DESIGNID/${des}/g" > /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/run_glm_${subj}.sh
			cat /data/p_02071/choice-maps/scripts/designs/run_design | sed "s/SUBJID/${subj}/g"  | sed "s/SESSID/${sess}/g" | sed "s/DESIGNID/${des}/g" > /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/run_design_sub-${subj}
		
			chmod a+rwx /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/run_glm_${subj}.sh
		
			condor_submit /data/pt_02071/choice-maps/imagingData/sub-${subj}/ses-${sess}/1stLevel/design_${des}/run_design_sub-${subj}
    	done
	done
done
