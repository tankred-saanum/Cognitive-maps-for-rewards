% Write text file for motion parameters
root        = '/vols/Scratch/mgarvert/ManyMaps/imagingData/Subj_';

subj=25;
for session = 1:2
for run = 1:4
try
mc = load([root,num2str(subj),'/session_',num2str(session),'/run_',num2str(run),'/preprocess_smooth.feat/mc/prefiltered_func_data_mcf.par']);
dlmwrite([root,num2str(subj),'/session_',num2str(session),'/run_',num2str(run),'/preprocess_smooth.feat/mc/prefiltered_func_data_mcf.txt'],mc,'delimiter','\t','precision',4)

catch
disp(['Subj ',num2str(subj),' session ', num2str(session), ' run ', num2str(run), ' was not converted'])
end
end
end
