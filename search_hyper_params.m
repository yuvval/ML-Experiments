function search_hyper_params(search_params, clean_junk_mutex_files_flag)
% function search_hyper_params(search_params)
% performs a (parfor) grid search on hyper params. use
% postprocess_search_hyper_params() to retrieve the results. this function
% has mutual exclusion mechnism, so it can be executed concurrently by different machines
% postprocess_search_hyper_params() can be called before the search execution
% terminates
% A NOTE REGARDING SEED: We use the given seed to select the validation
% subset. However, we change it later in order to choose a random sequence
% of training. Therefore EACH OF THE training procedures NEEDS TO SET THE SEED BY ITSELF!
%
% example for setting search_params struct:
%     search_params{k}.train_func = @train_func_handle; % syntax is [result_criterion, full_fname_results] = train_func(hyper_param_id, cfg_params, examples, labels, training_fold_logical_index);
%     search_params{k}.load_data_func = @load_dataset_func_handle; %syntax is [examples, labels, cfg_params] = load_dataset_func(cfg_params)
%     search_params{k}.cfg_params = cfg_params;
%     search_params{k}.dataset_fold = folds.training(k); % where folds = cvpartition(num_examples, 'KFold', kfolds);
%     search_params{k}.dataset_fold_id = k; % where folds = cvpartition(num_examples, 'KFold', kfolds);
%     search_params{k}.kfolds = 5;
%     search_params{k}.hyper_params_sweep;

if nargin<2
    clean_junk_mutex_files_flag = false;
end



%% load the dataset
[all_examples, all_labels, search_params.cfg_params] = search_params.load_data_func(search_params.cfg_params);

% take the entire subset for this (outer) fold
examples = all_examples(search_params.dataset_fold == 1, :);
if isfield(search_params.cfg_params, 'new_samples_inds')
    search_params.cfg_params.new_samples_inds = search_params.cfg_params.new_samples_inds(search_params.dataset_fold == 1);
end
if isvector(all_labels)
    %labels is a vector
    labels = all_labels(search_params.dataset_fold == 1);
else
    %labels is a matrix of weak similarity supervision
    labels = all_labels(search_params.dataset_fold == 1, search_params.dataset_fold == 1);
end
clear all_labels; % free up some memory
clear all_examples; % free up some memory
fprintf('search_hyper_params: after load the dataset, %s\n\n', get_mem_usage());

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
search_results_fnames = cell(size(search_results_criteria));
hyper_params = cell(size(search_results_criteria));

fold_ids = train_params_comb(end,:);

train_func = search_params.train_func;
cfg_params = search_params.cfg_params;
fname_func = search_params.train_results_fname_func;
ofold = search_params.dataset_fold_id;

for comb_id = 1:length(search_results_criteria)
    hyper_params{comb_id} = hyper_param_comb_to_struct(train_params_comb(:, comb_id), hyper_params_sweep);    
    ifold = fold_ids(comb_id);
    if clean_junk_mutex_files_flag % delete junk 'touch' file locks if exist
        results_filename = fname_func(cfg_params, hyper_params{comb_id}, ofold, ifold);
        full_fname_touch =  fullfile(cfg_params.path_results_mat , ['touch_', results_filename] );
        if exist(full_fname_touch, 'file')
            system(['rm ', full_fname_touch]); % removed the touched file
        end
    end
end
if clean_junk_mutex_files_flag  % delete all junk 'touch' file locks if exist and exit
    return; %exit
end

% select hyper params combinations to draw
rng(ceil(cputime*100)) % RANDOM seed to select hyper params combinations to draw
hyper_params_combinations_ids = randperm(length(search_results_criteria));
% parallel iterate on combination of hyper params and inner folds.
if ~isempty(regexp(version('-release'), '2014', 'match')) % command below is supported only for R2014
    isOpen = ~isempty(gcp('nocreate')); % If no pool, use regular for loop
else
    isOpen = false;
end
if isOpen
    parfor k = 1:length(search_results_criteria)
        comb_id = hyper_params_combinations_ids(k);
        ifold = fold_ids(comb_id);
        
        train_wrapper(train_func, fname_func, hyper_params{comb_id}, cfg_params, examples, labels, ifolds.training(ifold), ofold, ifold);
    end
else
    for k = 1:length(search_results_criteria)
        comb_id = hyper_params_combinations_ids(k);
        ifold = fold_ids(comb_id);
        
        train_wrapper(train_func, fname_func, hyper_params{comb_id}, cfg_params, examples, labels, ifolds.training(ifold), ofold, ifold);
    end
end



end


function [search_results_fnames] = train_wrapper(train_func, fname_func, hyper_params, cfg_params, examples, labels, training_fold_logical_index, ofold, ifold)
cfg_params.hyper_params_comb = hyper_params; % saving hyper params for later usage (for generating snapshots filenames)
results_filename = fname_func(cfg_params, hyper_params, ofold, ifold);
pause(rand()*5); % pause for a random time, before accessing the filesystem
full_fname_results = fullfile(cfg_params.path_results_mat ,[results_filename '.mat']);

if exist(full_fname_results, 'file') % if results file exists, then we skip this hp selection
    fprintf('skipping because exists: %s\n', results_filename);
else
    try
        pause(rand()*20); % pause for a random time, before accessing the filesystem
        full_fname_touch =  fullfile(cfg_params.path_results_mat , ['touch_', results_filename] );
        if ~exist(full_fname_touch, 'file')
            system(['touch ', full_fname_touch]); % removed the touched file
            [search_results_fnames] = ...
                train_func(hyper_params, cfg_params, examples, labels, training_fold_logical_index, ofold, ifold);
            system(['rm ', full_fname_touch]); % removed the touched file
        else
            fprintf('skipping because touched: %s\n', results_filename)
        end
        
    catch exception
        rm_cmd = ['rm ', full_fname_touch];
        system(rm_cmd); % removed the touched file
        disp(rm_cmd);
        
        rethrow(exception)             % Line 5
    end
end
end

