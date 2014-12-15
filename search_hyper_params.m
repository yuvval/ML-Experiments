function best_hyper_param_id = search_hyper_params(search_params)
% function best_hyper_param_id = search_hyper_params(search_params)
% returns best hyper param index following a grid search
% example for setting search_params struct:
%     search_params{k}.fun = @func_handle;
%     search_params{k}.load_data_func = @load_dataset;        
%     search_params{k}.cfg_params = cfg_params;
%     search_params{k}.dataset_fold = folds.training(k); % where folds = cvpartition(num_examples, 'KFold', kfolds);
%     search_params{k}.kfolds = 5;        
%     search_params{k}.hyper_params_indices;        

    %% load the dataset
    [all_examples, all_labels, cfg_params] = search_params.load_data_func(search_params.cfg_params);
    
    % take the subset for this fold
    examples = all_examples(search_params.dataset_fold);
    labels = all_labels(search_params.dataset_fold);
    num_examples = length(labels);
    
    %% Split to K folds (training / validation)
    folds = cvpartition(num_examples, 'KFold', search_params.kfolds);
    
    


    