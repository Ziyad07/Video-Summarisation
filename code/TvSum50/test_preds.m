function test_preds(datadir, feature)
load ./ydata-tvsum50_test_v1.mat; % load the dataset

videos_test = struct;
for i = 1:size(tvsum50,2)
    if strcmp(feature, 'Resnet')
        temp = load(datadir+string(tvsum50(i).video)+'.mat');
    else
        temp = load(datadir+string(tvsum50(i).video)+'_'+feature+'.mat');
    end
    
    if size(temp.x,2) < tvsum50(i).nframes 
        missing_frames = tvsum50(i).nframes - size(temp.x,1);
        zeross = zeros(1, missing_frames);
        temp.x = horzcat(temp.x', zeross); 
    end
    videos_test(i).Ypred = temp.x;
    videos_test(i).fileName = string(tvsum50(i).video);
end
[res] = evaluate_TVSum(videos_test, tvsum50, feature);
for j = 1: size(videos_test,2)
    f_measure(j) = res(j).mean_f1;
end
mean_res = mean(f_measure);
fprintf('f-score: %.3f\n', mean_res);
% fprintf('file name: %s f-score: %.3f\n', file_name, mean_res);

function [stats] = evaluate_TVSum(videos_test, tvsum50, feature)
budget = 0.15;
%% Normalize data
for i=1:length(videos_test)
    videos_test(i).Ypred = videos_test(i).Ypred - min(videos_test(i).Ypred);
    videos_test(i).frame_score = videos_test(i).Ypred / max(videos_test(i).Ypred);
end

%% Generate segmentation results
if strcmp(feature,'rgb')
    load(['../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/AllData_rgb_.mat']);
elseif strcmp(feature,'flow')
    load(['../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/AllData_flow_.mat']);
elseif strcmp(feature,'combined')
    load(['../../saved_numpy_arrays/TvSum50/predictions/I3D/MLP/AllData_flow_.mat']);
elseif strcmp(feature, 'Resnet')
    load(['../../saved_numpy_arrays/TvSum50/predictions/Resnet/MLP/AllData.mat']);
end

%load ('shot_TVSum.mat')
n_videos = [2,6,13,19,24,30,32,38,42,50];
seg_idx = cell(1,10);
for i=1:size(n_videos,2),
    for k=1:size(videos_test,2)
        if fileName(i,:) == videos_test(k).fileName
            shot_boundary = shot_boundaries{i};
            seg_tmp = zeros(numel(shot_boundary)-1,2);
            for j=1:numel(shot_boundary)-1,
                seg_tmp(j,2) = shot_boundary(j+1);
                seg_tmp(j,1) = shot_boundary(j)+1;
            end
            seg_idx{k} = seg_tmp;
        end
    end
end


teInds = zeros(1,size(tvsum50,2));
for i = 1:size(tvsum50,2)
    for j = 1:size(videos_test,2)
        if tvsum50(i).video == videos_test(j).fileName
            teInds(i) = j;
        end
    end
end

%% Evaluate performance
stats = struct;
% 50 videos
for i = teInds
%     disp(i)
    % Get prediction label
    pred_lbl = videos_test(i).frame_score;
    
    % Get predicted shot segments
    pred_seg = seg_idx{i};
    
    % Get ground-truth label
    gt_lbl = tvsum50(i).user_anno;
    
    % Compute pairwise f1 scores
    ypred = solve_knapsack( pred_lbl, pred_seg, budget );
    ytrue = cell(size(gt_lbl, 2),1);
    for j = 1:size(gt_lbl,2)
        ytrue{j} = solve_knapsack(gt_lbl(:,j), pred_seg, budget );
%         cp{j} = classperf( ytrue{j}, ypred, 'Positive', 1, 'Negative', 0 );
        cp{j} = classperf( ytrue{j}, ypred, 'Positive', 1, 'Negative', 0 );
    end
    stats(i).ypred = ypred;
    stats(i).video  = tvsum50(i).video;
    
    stats(i).prec = cellfun(@(x) x.PositivePredictiveValue, cp);
    stats(i).rec  = cellfun(@(x) x.Sensitivity, cp);
    stats(i).f1 = max(0, 2*(stats(i).prec.*stats(i).rec)./(stats(i).prec + stats(i).rec));
    
    stats(i).mean_f1 = sum(stats(i).f1) / size(gt_lbl,2);
end