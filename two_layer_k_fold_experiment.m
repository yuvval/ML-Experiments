function results = two_layer_k_fold_experiment(experiment_params, model_cfg_params, experiment_stage)
% function results = two_layer_k_fold_experiment(experiment_params, model_cfg_params, experiment_stage)
% Runs a 2 layer K folds experiment.
% input params: experiment params, model_configuration params
%               experiment_stage: either {'search_hyperparams'}, {'clean_junk_mutex'}
%               {'postprocess_search_hp', folds_range},
%               {'final_experiment', search_criterion_str}
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
%             experiment_params.train_results_fname_func = @train_results_fnames_by_hyper_param;
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
    if ~iscell(experiment_stage)
        experiment_stage = {experiment_stage};
    end
    
    
    %% load the dataset
    if ~strcmp(experiment_stage{1}, 'postprocess_search_hp')
        [examples, labels, model_cfg_params] = experiment_params.load_data_func(model_cfg_params);
        num_examples = length(labels);
    else
        dont_load_data_flag = 1;
        [~, ~, model_cfg_params] = experiment_params.load_data_func(model_cfg_params, dont_load_data_flag);
        num_examples = 666;
    end

    %% saving experiment and model configurations for results struct (for history)
    results.model_cfg_params = model_cfg_params;
    results.experiment_params = experiment_params;
    
    %% Split to K folds    
    rng('default')
    rng(hyper_params_sweep.seed) % make sure to sync the seed before a split
    
    % if split was 'manually' set by the load_data_func, then use that
    % split. Otherwise, generate a random split
    if iscell(model_cfg_params.split)
        ofolds = model_cfg_params.split;
        if kfolds ~= length(ofolds)
            error('kfolds and given split length should match')
        end
    else        
        ofolds_partition = cvpartition(num_examples, 'KFold', kfolds);
        for k=1:kfolds
            ofolds{k} = ofolds_partition.training(k);
        end
    end

    %% Grid search on hyper params
    [search_params, search_results, results_fnames, best_hyper_params] = deal(cell(kfolds, 1));
    for k=1:kfolds
        search_params{k}.train_func = experiment_params.train_func;
        search_params{k}.train_results_fname_func = experiment_params.train_results_fname_func;        
        search_params{k}.load_data_func = experiment_params.load_data_func;        
        search_params{k}.cfg_params = model_cfg_params;
        search_params{k}.dataset_fold = ofolds{k};
        search_params{k}.dataset_fold_id = k;
        search_params{k}.kfolds = 5;        
        search_params{k}.hyper_params_sweep = hyper_params_sweep;        
    end

    %               experiment_stage: either 'search_hyperparams',
%               'postprocess_search_hp', 'final_experiment'
    switch experiment_stage{1}
        case 'clean_junk_mutex'
            for k=1:kfolds
                search_hyper_params(search_params{k}, true);
            end
            fprintf('cleaned junk mutex files')
            
        case 'search_hyperparams'
            results.tstart = datestr(now);
            
            % To get better distribute the jobs over CPUs, we 
            % iterate on K outer folds in random order. always starting 1st fold
            rng(int32(mod(cputime,1)*100)); % random seed according to current cputime
            if length(experiment_stage) == 1
                folds_iter = randperm(kfolds);
            else
                folds_iter = experiment_stage{2}; % iterate on folds (or subset of them) in given order
            end                
            
            rng(hyper_params_sweep.seed); % syncing seed again
            
            % iterate on each outer fold and grid search on it
            for k=folds_iter
                fprintf('Search: Outer fold = %d\n', k');
                search_hyper_params(search_params{k});
            end
            results.tend = datestr(now);

            % postprocess_search_hp
            for k=1:folds_iter
                search_results{k} = postprocess_search_hyper_params( search_params{k} );
            end
            results.search_results = search_results;
        case 'postprocess_search_hp'
            for k=experiment_stage{2}
                search_results{k} = postprocess_search_hyper_params( search_params{k} );
            end
            results.search_results = search_results;
        case 'final_experiment'
            if length(experiment_stage)>2
                search_folds = experiment_stage{3};
            else
                search_folds = 1:kfolds;
            end
            for k=search_folds
                search_results{k} = postprocess_search_hyper_params( search_params{k} );
            end
            search_criterion = experiment_stage{2};
            criterion_id = find(ismember(search_criterion, search_results{search_folds(1)}.criteria_names),1);
            for k=1:kfolds
                % get best hyper param id
                if length(experiment_stage)>2 % get it from a single outer fold (to all other outer folds)
                    hp_comb_vector = search_results{experiment_stage{3}}.best_hyper_params_per_criterion(criterion_id, :);
                else
                    hp_comb_vector = search_results{k}.best_hyper_params_per_criterion(criterion_id, :);
                end
                best_hyper_params{k} = hyper_param_comb_to_struct(hp_comb_vector, hyper_params_sweep);
            end
            
            
            %% final experiment 
            % iter on each fold, train with best hyper param and prepare results
            % parallel iterate on combination of hyper params and inner folds.
            if length(experiment_stage)>3
                k = experiment_stage{4};
                fprintf('Experiment: Fold = %d\n', k');
                search_params{k}.train_func(best_hyper_params{k}, model_cfg_params, examples, labels, ofolds{k}, k, nan);
                return
            else
                parfor k=1:kfolds
                    fprintf('Experiment: Fold = %d\n', k');
                    [results_fnames{k}, result_criteria{k}] = search_params{k}.train_func(best_hyper_params{k}, model_cfg_params, examples, labels, ofolds{k}, k, nan);
                end
            end
            
            %% prepare results to return
            results.fnames = results_fnames;
            results.result_criteria = result_criteria;
            results.search_results = search_results;
            results.best_hyper_params = best_hyper_params;

            %% save results to a .mat file before returning
            results_fname = fullfile(experiment_params.path_results_mat, experiment_params.experiment_results_ref_fname);
            save(results_fname, 'results')
            fprintf('Saved experiment results on:\n %s\n', results_fname);
            
        otherwise
            disp('unknown experiment stage')
    end
    
    
            
   
    
end