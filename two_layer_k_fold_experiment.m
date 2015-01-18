function results = two_layer_k_fold_experiment(experiment_params, model_cfg_params)
% function results = two_layer_k_fold_experiment(experiment_params, model_cfg_params)
% Runs a 2 layer K folds experiment.
% input params: experiment params, model_configuration params
% returns: results structure with the following fields:
%             results.fnames = filenames cell array of each fold trial (with the best hyper param)
%             results.best_hyprm_id = best_hyprm_id for each fold 
%             results.criteria = criteria array for each fold
%             results.best_hyprm_max_steps_num = best_hyprm_max_steps_num during hyper param grid search (this value is used for the training of the outer fold experiment with best hyper param)
%             results.search_results = cell array with the results history of hyper param grid search
%
% usage example:
% 
%             experiment_params.load_data_func = @load_dataset; % syntax: [examples, labels, model_cfg_params] = load_dataset(model_cfg_params)
%             experiment_params.train_func = @similarity_train; % syntax: [result_criterion, full_fname_results, steps_num] = similarity_train(hyper_param_id, model_cfg_params, examples, labels, training_fold_logical_index, ofold, ifold, steps_num)
%             experiment_params.kfolds = kfolds;
%             experiment_params.hyper_params_sweep = hyper_params_sweep to iterate upon
%             experiment_params.experiment_results_ref_fname = ['similarity_experiment_' method_params_str '_' dataset_params_str '_githash_' git_commit_hash];
%             experiment_params.path_results_mat = '/cortex/yuvval/results_mat/experiments';
%             % additional non mandatory params (for history)
%             experiment_params.git_commit_hash = git_commit_hash;
%
%             set model_cfg_params according to the requirements of your similarity_training function
%


    %% init
    kfolds = experiment_params.kfolds;
    hyper_params_sweep = experiment_params.hyper_params_sweep;

    %% saving experiment and model configurations for results struct (for history)
    results.model_cfg_params = model_cfg_params;
    results.experiment_params = experiment_params;
    results.tstart = datestr(now);
    
    %% load the dataset
    [examples, labels, model_cfg_params] = experiment_params.load_data_func(model_cfg_params);
    num_examples = length(labels);

    %% Split to K folds
    rng(hyper_params_sweep.seed) % make sure to sync the seed before an split
    ofolds = cvpartition(num_examples, 'KFold', kfolds);

    %% Grid search on hyper params
    [search_params, search_history, results_fnames] = deal(cell(kfolds, 1));
    for k=1:kfolds
        search_params{k}.train_func = experiment_params.train_func;
        search_params{k}.load_data_func = experiment_params.load_data_func;        
        search_params{k}.cfg_params = model_cfg_params;
        search_params{k}.dataset_fold = ofolds.training(k);
        search_params{k}.dataset_fold_id = k;
        search_params{k}.kfolds = 5;        
        search_params{k}.hyper_params_sweep = hyper_params_sweep;        
    end
    
    [best_hyprm_id, best_hyprm_max_steps_num] = deal(zeros(kfolds, 1));
    for k=1:kfolds
        fprintf('Search: Outer fold = %d\n', k');
        [best_hyprm_id(k),best_hyprm_max_steps_num(k), search_history{k}] = search_hyper_params(search_params{k});
    end
    
    %% iter on each fold, train with best hyper param and prepare results
    model_cfg_params.train_aim = nan; % train_aim is used to indicate the train func the context it is running it: nan is the default. 'hpsearch' means that we run the training for hyper_param grid search.
    criteria_exp = nan(kfolds, 1);
    parfor k=1:kfolds
        
        fprintf('Experiment: Fold = %d\n', k');
        [criteria_exp(k), results_fnames{k}] = search_params{k}.train_func(best_hyprm_id(k), model_cfg_params, examples, labels, ofolds.training(k), k, nan, best_hyprm_max_steps_num(k));
    end
    
    %% prepare results to return
    results.fnames = results_fnames;
    results.criteria = criteria_exp;
    results.best_hyprm_id = best_hyprm_id;
    results.best_hyprm_max_steps_num = best_hyprm_max_steps_num;
    results.search_results = search_history;
    results.tend = datestr(now);

    %% save results to a .mat file before returning
    results_fname = fullfile(experiment_params.path_results_mat, experiment_params.experiment_results_ref_fname);
    save(results_fname, 'results')
    fprintf('Saved experiment results on:\n %s\n', results_fname);
            
   
    
end