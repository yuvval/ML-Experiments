function [ processed_search_results ] = postprocess_search_hyper_params( search_params )
% function [ processed_search_results ] = postprocess_search_hyper_params( search_params )
% returns:
%
% processed_search_results.train_params_comb = train_params_comb; % save all combs
% processed_search_results.results_criteria_mat = results_criteria_mat; % postprocessed results (each column is per a different combination)
% processed_search_results.results_criteria_mat = results_criteria_mat;
% processed_search_results.results = results;
% processed_search_results.mean_criteria_per_hp = mean_criteria_per_hp;
% processed_search_results.std_criteria_per_hp = std_criteria_per_hp;
% processed_search_results.hyper_params_combs = hyper_params_combs;
% processed_search_results.best_hyper_params_per_criterion = best_hyper_params_per_criterion;
% processed_search_results.criteria_names = criteria_names;
% processed_search_results.ofold = ofold;
% processed_search_results.valid_search_results = valid_search_results;
% processed_search_results.search_results_fnames = search_results_fnames;
% processed_search_results.results = results;
% processed_search_results.hyper_params = hyper_params; 
% processed_search_results.hyper_params_fieldnames = fieldnames(hyper_params_sweep); % save all combs fields names
% processed_search_results.hyper_params_sweep = hyper_params_sweep;
% processed_search_results.search_params = search_params;


%% prepare combinations of hyper_params and folds to train upon
hyper_params_sweep = search_params.hyper_params_sweep;
hp_fields_ranges = struct2cell(hyper_params_sweep);
train_params_comb = allcomb(hp_fields_ranges{1:end}, 1:search_params.kfolds).';


% init
valid_search_results  = zeros(size(train_params_comb,2),1);

search_results_fnames = cell(size(valid_search_results));
hyper_params          = cell(size(valid_search_results));
results               = cell(size(valid_search_results));

cfg_params = search_params.cfg_params;
fold_ids = train_params_comb(end,:);
ofold = search_params.dataset_fold_id;
fname_func = search_params.train_results_fname_func;

% iterate over all combs of hyperparams and inner folds
cnt_valid = 0;

n_criteria = nan; % init
criteria_names = {}; % init
results_criteria_mat = [];
for comb_id = 1:length(valid_search_results)    
    % aggregate hyper params
    hyper_params{comb_id} = hyper_param_comb_to_struct(train_params_comb(:, comb_id), hyper_params_sweep);
    ifold = fold_ids(comb_id);
    % aggregate fname to fnames cell array
    search_results_fnames{comb_id} = fname_func(cfg_params, hyper_params{comb_id}, ofold, ifold);

    % aggregate (loaded) results into a cell array struct
    try
        valid_search_results(comb_id) = 1;
        results{comb_id} = load(fullfile(cfg_params.path_results_mat ,search_results_fnames{comb_id}));
        cnt_valid = cnt_valid + 1;
        if cnt_valid == 1 %% init results_criteria_mat matrix
            n_criteria = numel(struct2cell(results{comb_id}.result_criteria));
            criteria_names = fieldnames(results{comb_id}.result_criteria);
            results_criteria_mat = nan(n_criteria, size(train_params_comb,2));
        end
        % aggregate possible search criteria results to a matrix, each column is a different comb
        results_criteria_mat(:, comb_id) = struct2array(results{comb_id}.result_criteria).';
    catch % if load is failed, mark this iteration as non valid and fill results with defaults        
        results_criteria_mat(:, comb_id) = nan;
        valid_search_results(comb_id) = 0;
        results{comb_id} = [];
    end
end

%% find mean of search criteria over inner folds of hyper params combinations
% get all combinations of hyper params (marginalizing inner folds)
hyper_params_combs = allcomb(hp_fields_ranges{1:end});
mean_criteria_per_hp = nan(n_criteria, size(hyper_params_combs,1));
std_criteria_per_hp  = nan(n_criteria, size(hyper_params_combs,1));
train_params_comb_cell = num2cell(train_params_comb,1); % conver rows to cell array

for hp_comb_id = 1:size(hyper_params_combs,1)
    hp_comb = hyper_params_combs(hp_comb_id, :);
    foo = @(v)(all(v(1:(end-1)) == hp_comb.'));
    hp_comb_indices = cellfun(foo, train_params_comb_cell);
    res_crit_hp = results_criteria_mat(:,hp_comb_indices).';
    mean_criteria_per_hp(:, hp_comb_id) = nanmean(res_crit_hp);    
    std_criteria_per_hp(:, hp_comb_id) = nanstd(res_crit_hp)./sqrt(sum(~isnan(res_crit_hp)));  % std of mean (s.e.m)  
end

if size(mean_criteria_per_hp,2) >1
    [~,maximizing_hp_ids] =  max(mean_criteria_per_hp.');
    [~, majority_hp_id] = max(hist(maximizing_hp_ids, 1:max(maximizing_hp_ids)));
else
    maximizing_hp_ids = 1;
    majority_hp_id = 1;
end
criteria_names{end+1} = 'majority_vote';
maximizing_hp_ids(end+1) = majority_hp_id;

best_hyper_params_per_criterion = hyper_params_combs(maximizing_hp_ids,:);
% majority_vote_hyper_params = hyper_params_combs(majority_hp_id,:);


processed_search_results.train_params_comb = train_params_comb; % save all combs
processed_search_results.results_criteria_mat = results_criteria_mat; % postprocessed results (each column is per a different combination)
processed_search_results.results_criteria_mat = results_criteria_mat;
processed_search_results.results = results;
processed_search_results.mean_criteria_per_hp = mean_criteria_per_hp;
processed_search_results.std_criteria_per_hp = std_criteria_per_hp;
processed_search_results.hyper_params_combs = hyper_params_combs;
processed_search_results.best_hyper_params_per_criterion = best_hyper_params_per_criterion;
processed_search_results.criteria_names = criteria_names;
% processed_search_results.majority_vote_hyper_params = majority_vote_hyper_params;
processed_search_results.ofold = ofold;
processed_search_results.valid_search_results = valid_search_results;
processed_search_results.search_results_fnames = search_results_fnames;
processed_search_results.results = results;
processed_search_results.hyper_params = hyper_params; 
processed_search_results.hyper_params_fieldnames = fieldnames(hyper_params_sweep); % save all combs fields names
processed_search_results.hyper_params_sweep = hyper_params_sweep;
processed_search_results.search_params = search_params;




end

