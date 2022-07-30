clear 
des = '106';
design = ['design_',des];
root = (['/data/pt_02071/choice-maps/imagingData/2ndLevel/',design]);
addpath(genpath('/data/p_02071/spm'));
wm = 4;

load (['/data/pt_02071/choice-maps/imagingData/sub-102/ses-2/1stLevel/',design,'/SPM.mat'])
load(['/data/p_02071/choice-maps/choice_data/population_data.mat']);
RL = all_data.RL;

sessiontype = {'2','3','both','diff'};

kernelsize = '8';

maxSubj=152;
savedir = ['/data/tu_garvert_cloud/owncloud-gwdg/Projects/ChoiceMaps/experiment/version_scan/datafiles/merged_data'];
load([savedir,'/population_data.mat'])
modeltype = 3; %1: unlimited values. 3: values bound between 0 and 100

clear winning_model aicsubj verrorsubj
% Individual subjects
kind = [repmat(2,52,1) repmat(3,52,6) ];
LLind = squeeze(-all_data.negLogLiklihood(:,modeltype,:));
AICind = 2*kind - 2*LLind;
AICind = AICind(:,[1 2 3 6 7]);
for subj = 1:52
    try
        winning_model(subj) = find(AICind(subj,:)==nanmin(AICind(subj,:)));
    catch
        winning_model(subj) = nan;
    end
end
winning_model(21)= nan;

clear ix
if length(wm) == 2
    group{1} = winning_model==wm(1);
    group{2} = winning_model==wm(2);
else
    group{1} = winning_model==wm(1);
    group{2} = winning_model~=wm(1);
end

h1 = figure; 
z = hist(winning_model,[1.01:1:17.01]);
bar(z)
hold on
xlim([0 size(AICind,2)*3+1])
prepImg
title('Winning model')


for i = 3:7
for session = 1:4
        try
            clear factorial_design matlabbatch
            if length(wm) > 1
                factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/groupCompare/',[num2str(wm(1)), '-',num2str(wm(2))],sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            else
                factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/groupCompare/',[num2str(wm(1))],sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            end
            delete([factorial_design.dir{1},'/*']);
            mkdir(factorial_design.dir{1})
            
            datadir= [root,'/session_',sessiontype{session},'/',sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            cd(datadir)
            
            contrast_images  = spm_select('List',['*_s',kernelsize,'_con*']);
            counter = 0;
            for epi = find(group{1})
                subjID = sprintf('1%02d',epi);
                if exist([datadir,'/',subjID,contrast_images(1,4:end)],'file')
                    counter = counter+1;
                    factorial_design.des.t2.scans1{counter,1} = [datadir,'/',subjID,contrast_images(1,4:end)];
                end
            end
            
            counter = 0;
            for epi = find(group{2})
                subjID = sprintf('1%02d',epi);
                if exist([datadir,'/',subjID,contrast_images(1,4:end)],'file')
                    counter = counter+1;
                    factorial_design.des.t2.scans2{counter,1} = [datadir,'/',subjID,contrast_images(1,4:end)];
                end
            end
            
            factorial_design.des.t2.dept = 0;
            factorial_design.des.t2.variance = 1;
            factorial_design.des.t2.gmsca = 0;
            factorial_design.des.t2.ancova = 0;
            
            factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.masking.tm.tm_none = 1;
            factorial_design.masking.im = 1;
            factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};
            factorial_design.globalc.g_omit = 1;
            factorial_design.globalm.gmsca.gmsca_no = 1;
            factorial_design.globalm.glonorm = 1;
            
            clear matlabbatch
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
            
            copyfile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_s',kernelsize,'_session_',sessiontype{session},'_0001_WM_groups.nii'])
             catch
        end
        saveas(h1,[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_s',kernelsize,'_session_',sessiontype{session},'_0001_WM_groups'],'png')
       
end

end


%%

for session = 1:4
    for g = 1:2
        try
            clear factorial_design matlabbatch
            if length(wm) > 1
                factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/group',num2str(g),'/',[num2str(wm(1)), '-',num2str(wm(2))],sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            else
                factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/group',num2str(g),'/',[num2str(wm(1))],sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            end
            delete([factorial_design.dir{1},'/*']);
            mkdir(factorial_design.dir{1})
            
            datadir= [root,'/session_',sessiontype{session},'/',sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            cd(datadir)
            
            contrast_images  = spm_select('List',['*_s',kernelsize,'_con*']);
            counter = 0;
            for epi = find(group{g})
                subjID = sprintf('1%02d',epi);
                if exist([datadir,'/',subjID,contrast_images(1,4:end)],'file')
                    counter = counter+1;
                    factorial_design.des.t1.scans{counter,1} = [datadir,'/',subjID,contrast_images(1,4:end)];
                end
            end
                                    
            factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.masking.tm.tm_none = 1;
            factorial_design.masking.im = 1;
            factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};
            factorial_design.globalc.g_omit = 1;
            factorial_design.globalm.gmsca.gmsca_no = 1;
            factorial_design.globalm.glonorm = 1;
            
            clear matlabbatch
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
            
            copyfile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_s',kernelsize,'_session_',sessiontype{session},'_0001_WM_group',num2str(g),'.nii'])
             catch
        end
    end
end

%%

if des=='137'
   for session = 1:4
        try
            clear factorial_design matlabbatch
            factorial_design.dir{1} = [root,'/session_',sessiontype{session},'/137_and_138/',sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            mkdir(factorial_design.dir{1})
            delete([factorial_design.dir{1} '/SPM.mat']);
            
            root = (['/data/pt_02071/choice-maps/imagingData/2ndLevel/design_137']);
            datadir= [root,'/session_',sessiontype{session},'/',sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            cd(datadir)
            
            contrast_images  = spm_select('List',['*_s',kernelsize,'_con*']);
            counter = 0;
            for epi = group{1}
                counter = counter+1;
                subjID = sprintf('1%02d',epi);
                factorial_design.des.t1.scans{counter,1} = [datadir,'/',subjID,contrast_images(1,4:end)];
            end
            
            root = (['/data/pt_02071/choice-maps/imagingData/2ndLevel/design_138']);
            datadir= [root,'/session_',sessiontype{session},'/',sprintf(['/%02d_',SPM.xCon(i).name,'_s',kernelsize],i)];
            cd(datadir)
            
            for epi = group{2}
                counter = counter+1;
                subjID = sprintf('1%02d',epi);
                factorial_design.des.t1.scans{counter,1} = [datadir,'/',subjID,contrast_images(1,4:end)];
            end
            
            factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
            factorial_design.masking.tm.tm_none = 1;
            factorial_design.masking.im = 1;
            factorial_design.masking.em = {'/data/pt_02071/choice-maps/imagingData/2ndLevel/brainmask/brainmask_thr.nii'};
            factorial_design.globalc.g_omit = 1;
            factorial_design.globalm.gmsca.gmsca_no = 1;
            factorial_design.globalm.glonorm = 1;
            
            clear matlabbatch
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
            
            copyfile([factorial_design.dir{1},'/spmT_0001.nii'],[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_s',kernelsize,'_session_',sessiontype{session},'_0001_WM_groups.nii'])
             catch
        end
        saveas(h1,[factorial_design.dir{1},'/',des,'_spmT_',SPM.xCon(i).name,'_s',kernelsize,'_session_',sessiontype{session},'_0001_WM_groups'],'png')
       
end

    
end

