
for subj = 101:152
    load([savedir,'/subj_',num2str(subj),'/data_',num2str(subj),'.mat']);
    for session = 2:3
        for run = 1:3
            try
                t1 = data.mat{session}.data.scan{run}.when_start;
                t2 = data.mat{session}.data.scan{run}.when_end;
                duration(subj-100,session-1,run) = datetime(t1)-datetime(t2);
            catch
                duration(subj-100,session-1,run) = nan;
            end
        end
    end
    try
        t1 = data.mat{3}.data.choice.start;
        t2 = data.mat{3}.data.choice.when_end;
        
        duration_choice(subj-100) = datetime(t1)-datetime(t2);
    catch
        duration_choice(subj-100) = nan;
    end
end
duration([21,36,37,38],:,:) = [];
duration_choice([21,36,37,38]) = [];

nanmean(duration)
nanmean(duration_choice)
nanstd(duration_choice)
