design = 'design_120';
root = (['/data/pt_02071/choice-maps/imagingData/2ndLevel/',design]);

load (['/data/pt_02071/choice-maps/imagingData/sub-102/ses-3/1stLevel/',design,'/SPM.mat'])

sessiontype = {'1','2','both','diff'};

maxSubj=152;
for i = 1:length(SPM.xCon)
    
for session = 2
    mkdir ([root,'/session_',num2str(session),sprintf(['/%02d_',SPM.xCon(i).name],i)])
    
    for subj = 101:maxSubj
        
        try
            copyfile (['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-',num2str(session),'/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)],...
                sprintf([root,'/session_',num2str(session),sprintf(['/%02d_',SPM.xCon(i).name],i),sprintf('/%02d_s_con_%04d.nii',subj,i)]));
        catch
            disp(['not successful for sub ',num2str(subj)])
        end
        
    end
    
    clear factorial_design matlabbatch
    
    factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/',sprintf(['%02d_',SPM.xCon(i).name],i)];
    cd(factorial_design.dir{1})
    
    delete([factorial_design.dir{1} '/SPM.mat']);
    contrast_images  = spm_select('List','*_s_con*');
    for epi = 1:size(contrast_images,1)
        factorial_design.des.t1.scans{epi,1} = [factorial_design.dir{1},'/',contrast_images(epi,:)];
    end
    factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    factorial_design.masking.tm.tm_none = 1;
    factorial_design.masking.im = 1;
    factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};
    factorial_design.globalc.g_omit = 1;
    factorial_design.globalm.gmsca.gmsca_no = 1;
    factorial_design.globalm.glonorm = 1;
    matlabbatch{1}.spm.stats.factorial_design = factorial_design;
    
    spm_jobman('run',matlabbatch);
    clear matlabbatch
    
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{2}.spm.stats.con.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'ttest';
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{2}.spm.stats.con.delete = 0;
    spm_jobman('run',matlabbatch);
    
    copyfile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_session_',sessiontype{session},'_0001.nii'])
    for i = 1:length(SPM.xCon)
        disp(sprintf(['---- Contrast %02d_',SPM.xCon(i).name,' -----'],i))
        mkdir([root,'/session_2/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        mkdir([root,'/session_3/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        mkdir([root,'/session_diff/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        mkdir([root,'/session_both/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        %     cd([root,'/session_diff/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        %     delete SPM.mat
        %     % Calculate difference between the two sessions
        %     for subj = 101:maxSubj
        %         if subj ~= 121 &amp;&amp; ~exist(sprintf('%s/session_diff/%02d_%s/%02d_s_con_%04d.nii',root,i,SPM.xCon(i).name,subj,i),'file')
        %             try
        %                 clear matlabbatch
        %                 matlabbatch{1}.spm.util.imcalc.input{1,1} = ['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-3/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)];
        %                 matlabbatch{1}.spm.util.imcalc.input{2,1} = ['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-2/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)];
        %                 matlabbatch{1}.spm.util.imcalc.output = sprintf('%02d_s_con_%04d.nii',subj,i);
        %                 matlabbatch{1}.spm.util.imcalc.outdir = {sprintf([root,'/session_diff/',sprintf(['%02d_',SPM.xCon(i).name],i)])};
        %                 matlabbatch{1}.spm.util.imcalc.expression = '(i1-i2)' ;
        %                 matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
        %                 matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
        %                 matlabbatch{1}.spm.util.imcalc.options.mask = 0;
        %                 matlabbatch{1}.spm.util.imcalc.options.interp = 1;
        %                 matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
        %                 spm_jobman('run',matlabbatch);
        %             catch
        %                 disp(['not successful for sub ',num2str(subj)])
        %             end
        %         end
        %     end
        %
        %     cd([root,'/session_both/',sprintf(['%02d_',SPM.xCon(i).name],i)])
        %     delete SPM.mat
        %     for subj = 101:maxSubj
        %         if subj ~= 121 &amp;&amp; ~exist(sprintf('%s/session_both/%02d_%s/%02d_s_con_%04d.nii',root,i,SPM.xCon(i).name,subj,i),'file')
        %             try
        %                 clear matlabbatch
        %                 matlabbatch{1}.spm.util.imcalc.input{1,1} = ['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-3/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)];
        %                 matlabbatch{1}.spm.util.imcalc.input{2,1} = ['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-2/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)];
        %                 matlabbatch{1}.spm.util.imcalc.output = sprintf('%02d_s_con_%04d.nii',subj,i);
        %                 matlabbatch{1}.spm.util.imcalc.outdir = {sprintf([root,'/session_both/',sprintf(['%02d_',SPM.xCon(i).name],i)])};
        %                 matlabbatch{1}.spm.util.imcalc.expression = '(i1+i2)/2' ;
        %                 matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
        %                 matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
        %                 matlabbatch{1}.spm.util.imcalc.options.mask = 0;
        %                 matlabbatch{1}.spm.util.imcalc.options.interp = 1;
        %                 matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
        %                 spm_jobman('run',matlabbatch);
        %             catch
        %                 disp(['not successful for sub ',num2str(subj)])
        %             end
        %         end
        %     end
        %
        for session = 2:3
            for subj = 101:maxSubj
                if ~(session == 3 &amp;&amp; subj == 121)
                    try
                        copyfile (['/data/pt_02071/choice-maps/imagingData/sub-',num2str(subj),'/ses-',num2str(session),'/1stLevel/',design,'/smooth/',sprintf('s_con_%04d.nii',i)],...
                            sprintf([root,'/session_',num2str(session),sprintf(['/%02d_',SPM.xCon(i).name],i),sprintf('/%02d_s_con_%04d.nii',subj,i)]));
                    catch
                        disp(['not successful for sub ',num2str(subj)])
                    end
                end
            end
        end
        for session = 1:4
            try
                clear factorial_design matlabbatch
                factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/',sprintf(['%02d_',SPM.xCon(i).name],i)];
                cd(factorial_design.dir{1})
                delete([factorial_design.dir{1} '/SPM.mat']);
                contrast_images  = spm_select('List','*_s_con*');
                for epi = 1:size(contrast_images,1)
                    factorial_design.des.t1.scans{epi,1} = [factorial_design.dir{1},'/',contrast_images(epi,:)];
                end
                factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
                factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
                factorial_design.masking.tm.tm_none = 1;
                factorial_design.masking.im = 1;
                factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};
                factorial_design.globalc.g_omit = 1;
                factorial_design.globalm.gmsca.gmsca_no = 1;
                factorial_design.globalm.glonorm = 1;
                matlabbatch{1}.spm.stats.factorial_design = factorial_design;
                spm_jobman('run',matlabbatch);
                clear matlabbatch
                matlabbatch{1}.spm.stats.fmri_est.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
                matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
                matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
                matlabbatch{2}.spm.stats.con.spmmat = {[factorial_design.dir{1} '/SPM.mat']};
                matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'ttest';
                matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = 1;
                matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
                matlabbatch{2}.spm.stats.con.delete = 0;
                spm_jobman('run',matlabbatch);
                copyfile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_session_',sessiontype{session},'_0001.nii'])
            catch
            end
        end
    end
end