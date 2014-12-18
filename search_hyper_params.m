function [best_hyprm_id, best_hyprm_max_steps_num, search_history] = search_hyper_params(search_params)
% function [best_hyprm_id, best_hyprm_max_steps_num, search_history] = search_hyper_params(search_params)
% returns best hyper param index following a grid search and results_fnames cell array
% example for setting search_params struct:
%     search_params{k}.train_func = @train_func_handle; % syntax is [result_criterion, full_fname_results] = train_func(hyper_param_id, cfg_params, examples, labels, training_fold_logical_index);
%     search_params{k}.load_data_func = @load_dataset_func_handle; %syntax is [examples, labels, cfg_params] = load_dataset_func(cfg_params)        
%     search_params{k}.cfg_params = cfg_params;
%     search_params{k}.dataset_fold = folds.training(k); % where folds = cvpartition(num_examples, 'KFold', kfolds);
%     search_params{k}.dataset_fold_id = k; % where folds = cvpartition(num_examples, 'KFold', kfolds);
%     search_params{k}.kfolds = 5;        
%     search_params{k}.hyper_params_indices;        


    %% set indicator that we train for searching hyper param
    search_params.cfg_params.train_aim = 'hpsearch';

    %% load the dataset
    [all_examples, all_labels, search_params.cfg_params] = search_params.load_data_func(search_params.cfg_params);
    
    % take the subset for this fold
    examples = all_examples(search_params.dataset_fold, :);
    labels = all_labels(search_params.dataset_fold);
    num_examples = length(labels);
    
    %% Split to K folds (training / validation)
    ifolds = cvpartition(num_examples, 'KFold', search_params.kfolds);
    
    %% prepare combinations of hyper_params and folds to train upon
    
    train_params_comb = allcomb(search_params.hyper_params_indices, 1:search_params.kfolds).';
    
    %% iterate (train) on all hyper_params and folds
    
    % preparing for parfor loop
    search_results_criteria = zeros(size(train_params_comb,2),1);
    search_results_steps_num = zeros(size(train_params_comb,2),1);
    search_results_fnames = cell(size(search_results_criteria));
    search_results_ifolds = cell(size(search_results_criteria));
    search_results_hyprm_ids = cell(size(search_results_criteria));
    
    hyper_param_ids = train_params_comb(1,:);
    fold_ids = train_params_comb(2,:);

    train_func = search_params.train_func;
    cfg_params = search_params.cfg_params;
    ofold = search_params.dataset_fold_id;
    parfor comb_id = 1:length(search_results_criteria)
        hyper_param_id = hyper_param_ids(comb_id);        
        search_results_hyprm_ids{comb_id} = hyper_param_id;
        ifold = fold_ids(comb_id);
        search_results_ifolds{comb_id} = ifold;
        [search_results_criteria(comb_id), search_results_fnames{comb_id}, search_results_steps_num(comb_id)] = ...
            train_func(hyper_param_id, cfg_params, examples, labels, ifolds.training(ifold), ofold, ifold);
    end
    % preparing parfor search results to return
    search_history.fnames = search_results_fnames;
    search_history.ifolds = search_results_ifolds;
    search_history.hyprm_ids = search_results_hyprm_ids;
    search_history.steps_num = search_results_steps_num;


    
    %% Find best (maximizing) hyper param

    iter = 1; avg_crit_per_hp = zeros(size(search_params.hyper_params_indices));
    for id = search_params.hyper_params_indices
        avg_crit_per_hp(iter) = mean(search_results_criteria(train_params_comb(1,:) == id));
        iter = iter+1;
    end
    best_hyprm_id = search_params.hyper_params_indices(avg_crit_per_hp==max(avg_crit_per_hp));   
    
    if (length(best_hyprm_id)) > 1 % if more than 1 hyparam gets best results we randomly choose one of the best
        best_hyprm_id = best_hyprm_id(randi(length(best_hyprm_id)));
    end
    
    best_hyprm_max_steps_num = max(search_results_steps_num(hyper_param_ids == best_hyprm_id));


    