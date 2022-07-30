addpath /vols/Scratch/mgarvert/ManyMaps/myPreprocessing/spm12/;
spm('defaults','fMRI');

GLMtoRun='XXjobidXX';
disp(GLMtoRun)

% for i=1:numel(inputs)
    fprintf([repmat('%%',1,72) '\n']);
    try
    	load(GLMtoRun);
        who
        
	fprintf('Design specification GLM %s\n',GLMtoRun);
       spm_jobman('run',design_specification);
		fprintf('Done\n');
        fprintf([repmat('%%',1,72) '\n']);
    
        fprintf('Estimating GLM %s\n',GLMtoRun);
        spm_jobman('run',estimation);
		fprintf('Done\n');

	fprintf('Running ttest GLM %s\n',GLMtoRun);
        spm_jobman('run',ttest);
                fprintf('Done\n');
        
    catch
        warning('Estimation failed.');
		warning(lasterr);
    end
    fprintf([repmat('%%',1,72) '\n']);
    save(GLMtoRun)
 % end


