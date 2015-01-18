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
%     search_params{k}.hyper_params_sweep;        


    %% load the dataset
    [all_examples, all_labels, search_params.cfg_params] = search_params.load_data_func(search_params.cfg_params);
    
    % take the subset for this fold
    examples = all_examples(search_params.dataset_fold, :);
    labels = all_labels(search_params.dataset_fold);
    num_examples = length(labels);
    
    %% Split to K folds (training / validation)
    rng(search_params.hyper_params_sweep.seed) % make sure to sync the seed before a split    
    ifolds = cvpartition(num_examples, 'KFold', search_params.kfolds);
    
    %% prepare combinations of hyper_params and folds to train upon
    hyper_params_sweep = search_params.hyper_params_sweep;
    hp_fields_ranges = struct2cell(hyper_params_sweep);
    train_params_comb = allcomb(hp_fields_ranges{1:end}, 1:search_params.kfolds).';
    
    %% iterate (train) on all hyper_params and folds
    
    % preparing for parfor loop
    search_results_criteria = zeros(size(train_params_comb,2),1);
    search_results_steps_num = zeros(size(train_params_comb,2),1);
    search_results_fnames = cell(size(search_results_criteria));
    hyper_params = cell(size(search_results_criteria));
    
    fold_ids = train_params_comb(end,:);

    train_func = search_params.train_func;
    cfg_params = search_params.cfg_params;
    ofold = search_params.dataset_fold_id;
    
    % delete all junk 'touch' file locks if exist
    for comb_id = 1:length(search_results_criteria)
        hyper_params{comb_id} = hyper_param_comb_to_struct(train_params_comb(:, comb_id), hyper_params_sweep);
        ifold = fold_ids(comb_id);
        [~, results_filename, ~] = init_hyper_params(struct('cfg_params', cfg_params), hyper_params{comb_id}, ofold, ifold);
        full_fname_touch =  fullfile(cfg_params.path_results_mat , ['touch_', results_filename] );
        if exist(full_fname_touch, 'file')
            system(['rm ', full_fname_touch]) % removed the touched file
        end
    end
    
    % parallel iterate on combination of hyper params and inner folds. 
    for comb_id = 1:length(search_results_criteria)
        ifold = fold_ids(comb_id);
        [search_results_criteria(comb_id), search_results_fnames{comb_id}, search_results_steps_num(comb_id)] = ...
            train_wrapper(train_func, hyper_params{comb_id}, cfg_params, examples, labels, ifolds.training(ifold), ofold, ifold);
    end
    
    
    
end

function hyper_params_struct = hyper_param_comb_to_struct(comb, hyper_params_sweep)    
    hyper_params_fieldnames = fieldnames(hyper_params_sweep);
    for i = 1:numel(hyper_params_fieldnames)
        hyper_params_struct.(hyper_params_fieldnames{i}) = comb(i);
    end
end

function [search_results_criteria, search_results_fnames, search_results_steps_num] = train_wrapper(train_func, hyper_params, cfg_params, examples, labels, training_fold_logical_index, ofold, ifold)

[~, results_filename, ~] = init_hyper_params(struct('cfg_params', cfg_params), hyper_params, ofold, ifold);
try
    pause(mod(cputime,1)); % pause for a random time (<1 sec), before accessing the filesystem
    full_fname_touch =  fullfile(cfg_params.path_results_mat , ['touch_', results_filename] );
    if ~exist(full_fname_touch, 'file')
        system(['touch ', full_fname_touch]) % removed the touched file
        [search_results_criteria, search_results_fnames, search_results_steps_num] = ...
            train_func(hyper_params, cfg_params, examples, labels, training_fold_logical_index, ofold, ifold);
        system(['rm ', full_fname_touch]) % removed the touched file
    end
    
catch exception
    system(['rm ', full_fname_touch]) % removed the touched file
    rethrow(exception)             % Line 5
end
end

