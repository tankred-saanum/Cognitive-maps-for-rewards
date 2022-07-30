#!/bin/bash -l

#SCRATCH=${HOME}/Scratch

#export MCR_INHIBIT_CTF_LOCK=1


# Force bash as the executing shell.

#$ -S /bin/bash


template=/vols/Scratch/mgarvert/ManyMaps/imagingData/scripts/designs/run_GLM_template.m
des=302_fsl_
rm -rf contrast_commands.txt
for subj in 1;do
    for c in {1..49};do
        contrast=$(printf "%03d" $c)
	echo contrast
        designFile=/vols/Scratch/mgarvert/ManyMaps/imagingData/Subj_${subj}/2ndLevel/design_$des/con_$contrast/design.mat
        contrastscript=/vols/Scratch/mgarvert/ManyMaps/imagingData/Subj_${subj}/2ndLevel/design_${des}/con_$contrast/run_contrast.m
        echo /vols/Scratch/mgarvert/ManyMaps/imagingData/Subj_${subj}/session_${sess}/2ndLevel/design_${des}/con_$contrast/run_contrast.m

        cat $template | sed s@XXjobidXX@"${designFile}"@g > $contrastscript

        echo "matlab -nodesktop -nodisplay -nosplash \< $contrastscript" >> contrast_commands.txt
        
    done
done

fsl_sub -q veryshort.q -t contrast_commands.txt








