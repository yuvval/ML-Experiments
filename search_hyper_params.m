function [best_hyper_param_id, results_fnames] = search_hyper_params(search_params)
% function [best_hyper_param_id, results_fnames] = search_hyper_params(search_params)
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
    search_params.cfg_params.train_objective = 'hpsearch';

    %% load the dataset
    [all_examples, all_labels, search_params.cfg_params] = search_params.load_data_func(search_params.cfg_params);
    
    % take the subset for this fold
    examples = all_examples(search_params.dataset_fold, :);
    labels = all_labels(search_params.dataset_fold);
    num_examples = length(labels);
    
    %% Split to K folds (training / validation)
    folds = cvpartition(num_examples, 'KFold', search_params.kfolds);
    
    %% prepare combinations of hyper_params and folds to train upon
    
    train_params_comb = allcomb(1:search_params.kfolds, search_params.hyper_params_indices).';
    
    %% iterate (train) on all hyper_params and folds
    
    search_result_criteria = zeros(size(train_params_comb,2),1);
    % todo change to parfor
    for comb_id = size(train_params_comb,2)
        fold_id = train_params_comb(1,comb_id);
        hyper_param_id = train_params_comb(2,comb_id);        
        search_params.cfg_params.fold_id = fold_id;
        search_params.cfg_params.dataset_fold_id = search_params.dataset_fold_id;
        search_result_criteria(comb_id) = search_params.train_func(hyper_param_id, search_params.cfg_params, examples, labels, folds.training(fold_id));
    end
    
    %% Todo: find best hyper param
    
    best_hyper_param_id = -1;   


    