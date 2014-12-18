function results = two_layer_k_fold_experiment(experiment_params, model_cfg_params)
    %% init
    kfolds = experiment_params.kfolds;
    hyper_params_indices = experiment_params.hyper_params_indices;

    %% saving experiment and model configurations for results struct (for history)
    results.model_cfg_params = model_cfg_params;
    results.experiment_params = experiment_params;
    
    %% load the dataset
    [examples, labels, model_cfg_params] = experiment_params.load_data_func(model_cfg_params);
    num_examples = length(labels);

    %% Split to K folds
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
        search_params{k}.hyper_params_indices = hyper_params_indices;        
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

    %% save results to a .mat file before returning
    results_fname = fullfile(experiment_params.path_results_mat, experiment_params.experiment_results_ref_fname);
    save(results_fname, 'results')
            
   
    
end