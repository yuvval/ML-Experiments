function hyper_params_struct = hyper_param_comb_to_struct(comb, hyper_params_sweep)    
% function hyper_params_struct = hyper_param_comb_to_struct(comb, hyper_params_sweep)    
% takes a combination of hyper params (as a vector), and the
% struct that contains the hyper_params_sweep (for the field names)
% and generates a new struct with these hyper params

    hyper_params_fieldnames = fieldnames(hyper_params_sweep);
    for i = 1:numel(hyper_params_fieldnames)
        hyper_params_struct.(hyper_params_fieldnames{i}) = comb(i);
    end
end
